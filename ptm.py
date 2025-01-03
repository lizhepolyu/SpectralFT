import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


# -----------------------------------------------------
# 1. Load and Wrap the Pretrained Model
# -----------------------------------------------------
model = WavLMModel.from_pretrained("microsoft/wavlm-large")
model = model.cuda()

# If you need to freeze the model during inference,
# uncomment the following line:
# model.eval()


# -----------------------------------------------------
# 2. Define the SpectralAdapter Class
# -----------------------------------------------------
class SpectralAdapter(nn.Module):
    """
    Decompose an input weight using Singular Value Decomposition (SVD),
    then introduce a trainable low-rank perturbation to form a new weight matrix.

    Parameters:
    ----------
    initial_weight: torch.Tensor
        The original weight matrix to be adapted.
    rank: int
        The rank (r) used for truncated SVD.
    """
    def __init__(self, initial_weight: torch.Tensor, rank: int):
        super(SpectralAdapter, self).__init__()

        # The rank cannot exceed the smallest dimension of the original weight.
        assert rank <= min(initial_weight.size()), (
            "Rank r cannot exceed the dimensions of the initial weight."
        )

        # Perform SVD decomposition on the initial_weight (without full_matrices).
        U, S, Vh = torch.linalg.svd(initial_weight, full_matrices=False)

        # Truncate to the specified rank.
        self.U1 = U[:, :rank].detach()
        self.S  = torch.diag(S[:rank]).detach()
        self.Vh1 = Vh[:rank, :].detach()

        # Define the dimension of the low-rank perturbation (can be customized).
        low_rank = 16
        input_dim = initial_weight.shape[1]

        # Define trainable parameters: low-rank increments for U and Vh.
        self.UA = nn.Parameter(torch.randn(input_dim, low_rank))
        self.UB = nn.Parameter(torch.randn(low_rank, rank))

        self.VhA = nn.Parameter(torch.randn(rank, low_rank))
        self.VhB = nn.Parameter(torch.randn(low_rank, input_dim))

        # Initialize the trainable parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the trainable parameters.
        """
        nn.init.kaiming_uniform_(self.UA)
        nn.init.zeros_(self.UB)
        nn.init.kaiming_uniform_(self.VhA)
        nn.init.zeros_(self.VhB)

    def forward(self) -> torch.Tensor:
        """
        Compute and return the updated weight matrix.
        """
        # Update U and Vh with the low-rank perturbation.
        U_new = self.U1 + torch.matmul(self.UA, self.UB)
        Vh_new = self.Vh1 + torch.matmul(self.VhA, self.VhB)

        # Reconstruct the final weight matrix.
        updated_weight = torch.matmul(torch.matmul(U_new, self.S), Vh_new)
        return updated_weight


# -----------------------------------------------------
# 3. Define the WavLMPtm Class
# -----------------------------------------------------
class WavLMPtm(nn.Module):
    """
    A wrapper for the WavLM model. It applies spectral adaptation to the attention
    weights (q_proj and k_proj) of each encoder layer, and provides a weighted 
    combination of the hidden states from all layers.

    Usage:
    ------
    - Frozen the main WavLM parameters (no gradient updates on them).
    - Each encoder layer's attention weights (q_proj, k_proj) are adapted with
      an independent SpectralAdapter.
    - The final output is a weighted sum of layer-wise representations, 
      passed through optional instance normalization (or other transformations).
    """
    def __init__(self):
        super(WavLMPtm, self).__init__()
        
        # WavLM-large has 24 encoder layers plus 1 embedding layer => total 25
        num_layers = model.config.num_hidden_layers + 1

        # Initialize learnable softmax-based weights for all hidden_states
        initial_weights = torch.randn(num_layers)
        self.weights = nn.Parameter(F.softmax(initial_weights, dim=0), requires_grad=True)

        # Set the rank for truncated SVD
        self.rank = 256

        # Example instance normalization on feature dimension 1024
        self.instance_norm = nn.InstanceNorm1d(1024)

        # Freeze all pretrained model parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Build separate SpectralAdapters for each layer's q_proj and k_proj
        self.spectral_adapters_q = nn.ModuleDict()
        self.spectral_adapters_k = nn.ModuleDict()

        for i in range(model.config.num_hidden_layers):
            q_key = f'encoder_layers_{i}_attention_q_proj_weight'
            k_key = f'encoder_layers_{i}_attention_k_proj_weight'

            q_weight = getattr(model.encoder.layers[i].attention, 'q_proj').weight
            k_weight = getattr(model.encoder.layers[i].attention, 'k_proj').weight

            self.spectral_adapters_q[q_key] = SpectralAdapter(q_weight, self.rank)
            self.spectral_adapters_k[k_key] = SpectralAdapter(k_weight, self.rank)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Update q_proj and k_proj weights in each layer's attention using SpectralAdapters.
          2. Forward the audio through the WavLM model to get hidden states for each layer.
          3. Compute a weighted sum of those hidden states.
          4. (Optional) Apply instance normalization or any other processing.
          5. Return the final representation.

        Args:
        -----
        audio: torch.Tensor
            The input audio tensor, shape (B, T) or (B, 1, T).
            Make sure the shape is correct before passing in.

        Returns:
        --------
        x: torch.Tensor
            The final feature tensor with shape (B, C, T), typically C=1024.
        """
        # Step 1: Update q_proj and k_proj via SpectralAdapters (frozen model weights are updated in-place)
        for i in range(model.config.num_hidden_layers):
            q_key = f'encoder_layers_{i}_attention_q_proj_weight'
            k_key = f'encoder_layers_{i}_attention_k_proj_weight'

            q_adapter_weight = self.spectral_adapters_q[q_key]()
            k_adapter_weight = self.spectral_adapters_k[k_key]()

            model.encoder.layers[i].attention.q_proj.weight = nn.Parameter(q_adapter_weight)
            model.encoder.layers[i].attention.k_proj.weight = nn.Parameter(k_adapter_weight)

        # (Optional) layer normalization on the input audio
        audio = F.layer_norm(audio, audio.shape)

        # Step 2: Forward pass through WavLM, retrieving all hidden states
        outputs = model(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # A tuple of hidden states from each layer

        # Step 3: Weighted sum of layer-wise representations
        # Note that self.weights is also normalized by softmax
        weighted_sum = sum(w * rep for w, rep in zip(torch.softmax(self.weights, dim=0), hidden_states))

        # Step 4: Permute from (B, T, C) -> (B, C, T) if needed
        x = weighted_sum.permute(0, 2, 1)

        # Example: Instance normalization if using HuBERT or similar
        # x = x + 1e-6
        # x = self.instance_norm(x)

        # Step 5: Return the final representation
        return x

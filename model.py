"""
ECAPA-TDNN Model

This implementation is a combination of ideas from the following projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptm import WavLMPtm


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) Module:
    Applies a learnable gating mechanism to emphasize useful feature channels.

    Args:
        channels (int): Number of input/output feature channels.
        bottleneck (int): Intermediate dimensionality within the SE block.
    """
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # Optional BatchNorm can be added here if needed
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass: Multiply the original input by the learned channel-wise mask.
        """
        se_mask = self.se(x)
        return x * se_mask


class Bottle2neck(nn.Module):
    """
    Bottleneck Res2Net block with scale-based splitting and dilation.

    This structure splits incoming features into multiple chunks and applies
    convolutions with a specified kernel size and dilation. The feature chunks
    are then concatenated back to form the output. An SE module is applied at the end.

    Args:
        inplanes (int): Number of input feature channels.
        planes (int): Number of output feature channels.
        kernel_size (int): Kernel size for the intermediate convolutions.
        dilation (int): Dilation rate for the intermediate convolutions.
        scale (int): Number of splits for 'res2' style grouping.
    """
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))

        # Initial 1x1 convolution
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1
        self.width = width

        # Create multiple convolution paths
        convs, bns = [], []
        pad_size = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=pad_size))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # Final 1x1 convolution
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        # Activation and Squeeze-and-Excitation
        self.relu = nn.ReLU()
        self.se = SEModule(planes)

    def forward(self, x):
        """
        Forward pass of the Bottle2neck block.

        Steps:
          1. Convolution + split into chunks
          2. Convolution on each chunk + partial residual
          3. Concatenate outputs, 1x1 conv, SE, residual addition
        """
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        # Split the output into 'scale' chunks
        spx = torch.split(out, self.width, dim=1)
        running = None

        # Perform conv on each chunk, accumulate outputs
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            running = sp if i == 0 else torch.cat((running, sp), dim=1)

        # Concatenate the last chunk
        out = torch.cat((running, spx[self.nums]), dim=1)

        # Final 1x1 convolution
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        # Squeeze-and-Excitation and residual connection
        out = self.se(out)
        out += residual
        return out


class SpecAugment(nn.Module):
    """
    SpecAugment module for time-frequency masking.

    This module can randomly mask ranges in the time or frequency dimension
    to improve model robustness against variations.

    Args:
        freq_mask_width (tuple): The [min, max] width range for frequency masking.
        time_mask_width (tuple): The [min, max] width range for time masking.
    """
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width

    def mask_along_axis(self, x, dim):
        """
        Masks a random continuous range on the specified axis (time or frequency).

        Args:
            x (torch.Tensor): Feature map of shape (B, F, T).
            dim (int): Dimension to mask (1 for freq, 2 for time).
        """
        original_size = x.shape
        batch_size, n_freq, n_time = x.shape

        if dim == 1:
            D = n_freq
            width_range = self.freq_mask_width
        else:
            D = n_time
            width_range = self.time_mask_width

        # Randomly choose mask lengths and start positions
        mask_len = torch.randint(width_range[0], width_range[1], (batch_size, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch_size, 1), device=x.device).unsqueeze(2)

        # Create a mask of shape (batch_size, D)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) & (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        # Expand mask along the other dimension
        if dim == 1:
            mask = mask.unsqueeze(2)  # (B, F, 1)
        else:
            mask = mask.unsqueeze(1)  # (B, 1, T)

        # Apply the mask
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        """
        Apply frequency masking then time masking in sequence.
        """
        x = self.mask_along_axis(x, dim=2)  # Time masking
        x = self.mask_along_axis(x, dim=1)  # Frequency masking
        return x


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN Speaker Encoder with integrated WavLM-based frontend.

    The architecture includes:
      - A WavLMPtm module for feature extraction (output dim=1024).
      - SpecAugment for data augmentation.
      - Multiple Res2Net bottleneck layers with Squeeze-and-Excitation.
      - An attentive statistic pooling layer (mixing mean & variance).
      - A final linear projection to a 192-dimensional embedding vector.

    Args:
        C (int): Channel dimension for the bottleneck layers.
    """
    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()

        # WavLM-based front-end
        self.WavLMPtm = WavLMPtm()

        # SpecAugment for optional data augmentation
        self.specaug = SpecAugment()

        # Initial projection from 1024-dim WavLM output to C channels
        self.conv1 = nn.Conv1d(1024, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        # Res2Net-based bottleneck layers
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)

        # Multi-scale feature aggregation layer (MFA)
        # from concatenated feature maps => 1536 channels
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        # Attentive Statistical Pooling
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # Additional non-linearity
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )

        # Final projection to 192-dim
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug):
        """
        Forward pass of the ECAPA-TDNN speaker encoder.

        Steps:
          1. Audio input => WavLM-based frontend => 1024-dim features.
          2. Optional SpecAugment (if aug=True).
          3. Series of Res2Net Bottleneck layers with SE.
          4. Multi-scale feature concatenation.
          5. Attentive Statistical Pooling (mean & std).
          6. Final linear + BN => 192-dim embedding.

        Args:
            x (torch.Tensor): Input waveform(s). Shape (B, T) or (B, 1, T).
            aug (bool): Whether to apply SpecAugment.

        Returns:
            torch.Tensor: Speaker embedding of shape (B, 192).
        """
        # Step 1: Extract features using WavLM
        x = self.WavLMPtm(x)

        # Step 2: Apply SpecAugment if enabled
        if aug:
            x = self.specaug(x)

        # Step 3: Convolution + ReLU + BN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Res2Net layers (bottleneck + SE)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        # Multi-scale feature aggregation
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.layer4(x)
        x = self.relu(x)

        # Step 4: Attentive Statistical Pooling
        t = x.size(-1)
        # Compute global mean & std
        mean = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t)
        std = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)

        # Concatenate local & global features
        global_x = torch.cat((x, mean, std), dim=1)
        w = self.attention(global_x)

        # Weighted mean and weighted standard deviation
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat([mu, sg], dim=1)
        x = self.bn5(x)

        # Step 5: Final linear projection + batch norm
        x = self.fc6(x)
        x = self.bn6(x)

        return x
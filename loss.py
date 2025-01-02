"""
AAMsoftmax Loss Function
Originally adapted from voxceleb_trainer:
https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import accuracy


class AAMsoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-softmax) implementation.

    This class computes the AAM-softmax loss for speaker classification tasks. It
    is commonly used for tasks involving face/speaker recognition where angular
    margins provide sharper decision boundaries.

    Parameters
    ----------
    n_class : int
        Number of output classes (e.g., the number of speakers).
    m : float
        Angular margin value.
    s : float
        Feature scale factor (often referred to as 'scale' or 'temperature').
    """
    def __init__(self, n_class: int, m: float, s: float):
        super(AAMsoftmax, self).__init__()

        self.m = m
        self.s = s

        # Learnable weight matrix for class centers
        # For a typical ECAPA-TDNN setup, each feature vector is of size 192
        self.weight = nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        # Cross-entropy loss for classification
        self.ce = nn.CrossEntropyLoss()

        # Precompute margin terms
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        """
        Forward pass for AAM-softmax.

        1. Normalize both input features 'x' and weight matrix 'self.weight'.
        2. Compute the cosine similarity between features and class weights.
        3. Calculate the additive angular margin.
        4. Scale and apply standard cross-entropy for speaker classification.

        Args
        ----
        x : torch.Tensor
            Input feature tensor of shape (batch_size, feature_dim).
        label : torch.Tensor, optional
            Class labels of shape (batch_size,). 

        Returns
        -------
        loss : torch.Tensor
            Computed AAM-softmax loss value.
        prec1 : float
            Top-1 accuracy for logging and analysis.
        """
        # Step 1: Normalize features and weights, then compute cosine similarity
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # Step 2: Compute 'phi' by applying angular margin
        #   sine = sqrt(1 - cosine^2), ensuring no negative values after clamp
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Conditionally apply margin
        #   If (cosine - th) <= 0, fallback to 'cosine - mm'
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # Step 3: Create one-hot encoding for labeling
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # Step 4: Combine margin-augmented logits ('phi') for target classes
        #         with original logits ('cosine') for non-target classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Step 5: Scale features before computing cross-entropy
        output = output * self.s

        # Step 6: Compute loss and top-1 accuracy
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
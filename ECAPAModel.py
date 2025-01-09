"""
This script contains logic to train the ECAPA-TDNN-based speaker model 
and evaluate its performance on a verification task.
"""

# -----------------------------
# 1. Standard Library Imports
# -----------------------------
import os
import sys
import time
import pickle
import random
from glob import glob

# -----------------------------
# 2. Third-Party Imports
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile
import tqdm
from scipy import signal
from itertools import chain

# -----------------------------
# 3. Local Imports
# -----------------------------
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN


class ECAPAModel(nn.Module):
    """
    ECAPA model wrapper that:
     1) Instantiates an ECAPA-TDNN backbone.
     2) Uses AAM-Softmax loss for speaker classification.
     3) Supports training, evaluation, and parameter management (save/load).
    """
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        """
        Args:
            lr (float): Initial learning rate for the optimizer.
            lr_decay (float): Learning rate decay factor (used by StepLR if enabled).
            C (int): Channel size for ECAPA-TDNN.
            n_class (int): Number of speaker classes for classification.
            m (float): Margin parameter for AAM-softmax.
            s (float): Scale parameter for AAM-softmax.
            test_step (int): Step interval for certain schedulers or evaluation steps.
            **kwargs: Additional keyword arguments (not directly used here).
        """
        super(ECAPAModel, self).__init__()

        # -------------------
        # 1. Define Model 
        # -------------------
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()

        # -------------------
        # 2. Define Loss 
        # -------------------
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()

        # -----------------------------
        # 3. Define Optimizer/Scheduler
        # -----------------------------
        # Using Adam optimizer and CosineAnnealingWarmRestarts as an example.
        # self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=40, T_mult=2, eta_min=1e-7)

        # Alternative (commented out) for reference:
        self.optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1.0e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)

        # Print the total number of model parameters (in MB)
        model_params = sum(
        param.numel() 
        for param in self.speaker_encoder.parameters() 
        if param.requires_grad
        )
        current_time = time.strftime("%m-%d %H:%M:%S")
        print(f"{current_time} Model parameter count = {model_params / 1024 / 1024:.2f} MB")
        
        # Assuming `wavlm_ptm` is an already initialized instance of the `WavLMPtm` class.
        adapters = chain(self.speaker_encoder.WavLMPtm.spectral_adapters_q.values(),
                        self.speaker_encoder.WavLMPtm.spectral_adapters_k.values())
                        
        spectral_adapter_params = sum(
            p.numel()
            for adapter in adapters          # adapter 是一个 SpectralAdapter 实例
            for p in adapter.parameters()    # p 是 adapter 内部真正的可训练张量
            if p.requires_grad
        )

        print(f"Trainable params in SpectralAdapters: {spectral_adapter_params / 1e6:.2f} M")
    
    def train_network(self, epoch, loader):
        """
        Perform one epoch of training.

        Args:
            epoch (int): Current epoch number (used by the scheduler).
            loader (DataLoader): PyTorch DataLoader returning (audio_data, labels) batches.

        Returns:
            average_loss (float): Mean training loss for the epoch.
            current_lr (float): Learning rate used.
            accuracy (float): Training accuracy within this epoch.
        """
        self.train()
        # Update the learning rate scheduler
        self.scheduler.step(epoch - 1)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Retrieve the initial learning rate for logging
        current_lr = self.optim.param_groups[0]['lr']

        # Iterate over the training DataLoader
        for batch_idx, (data, labels) in enumerate(loader, start=1):
            # Reset gradients
            self.zero_grad()

            # Prepare labels and audio data on GPU
            labels = torch.LongTensor(labels).cuda()
            data = data.cuda()

            # Forward pass through the speaker encoder
            feature = self.speaker_encoder.forward(data, aug=True)

            # Compute speaker classification loss and number of correctly classified samples
            nloss, correct = self.speaker_loss(feature, labels)

            # Backpropagation
            nloss.backward()
            self.optim.step()

            # Accumulate metrics
            total_loss += nloss.item()
            total_correct += correct.item()
            total_samples += labels.size(0)

            # Optional logging
            total_steps = len(loader)
            if batch_idx % 10 == 0 or batch_idx == total_steps:
                accuracy = total_correct / total_samples*len(labels)
                avg_loss = total_loss / batch_idx
                progress = 100.0 * batch_idx / total_steps
                current_time = time.strftime("%m-%d %H:%M:%S")

                # Print progress to stderr (allows partial printing without newline)
                sys.stderr.write(
                    f"{current_time} [Epoch {epoch:2d}] LR: {current_lr:.6f}, "
                    f"Progress: {progress:.2f}%, Loss: {avg_loss:.5f}, Acc: {accuracy:.2f}%\r"
                )
                # Flush less frequently for performance reasons
                if (batch_idx % 10 == 0 or batch_idx == total_steps) and batch_idx % 50 == 0:
                    sys.stderr.flush()

        # Return epoch metrics
        average_loss = total_loss / batch_idx
        accuracy =  total_correct / total_samples*len(labels)
        return average_loss, current_lr, accuracy


    def eval_network(self, eval_list_path, eval_data_path):
        """
        Evaluate the model for speaker verification. Embeddings are extracted for each audio file,
        then verification scores are computed on pairs.

        Args:
            eval_list (str): Path to a list of pairs [label, file1, file2].
            eval_path (str): Directory where the audio files are located.

        Returns:
            EER (float): Equal error rate.
            minDCF (float): Minimum detection cost function value.
        """
        self.eval()

        # Step 1. Collect all unique filenames from the eval list.
        files = []
        embeddings = {}
        lines = open(eval_list_path).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = sorted(list(set(files)))

        # Step 2. Compute embeddings for each audio file.
        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio_path = os.path.join(eval_data_path, file)
            audio, sr = soundfile.read(audio_path)
            
            # If the audio length is greater than 60 seconds, truncate to the first 60 seconds
            max_duration_sec = 60
            max_samples = int(sr * max_duration_sec)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Convert to FloatTensor for single-pass embedding
            data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

            # Handle split utterance approach for robust embedding

            min_duration_sec = 3
            min_samples = int(sr * min_duration_sec)
            if len(audio) <= min_samples:
                shortage = min_samples - len(audio)
                audio = np.pad(audio, (0, shortage), mode='wrap')

            # Split the audio into chunks
            feats = []
            start_frames = np.linspace(0, len(audio) - min_samples, num=5)
            for start in start_frames:
                start = int(start)
                feats.append(audio[start:start + min_samples])
            feats = np.stack(feats, axis=0).astype(float)

            data_2 = torch.FloatTensor(feats).cuda()

            # Extract embeddings
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(x=data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=-1)

                embedding_2 = self.speaker_encoder.forward(x=data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=-1)

            # Store embeddings for later pairing
            embeddings[file] = [embedding_1, embedding_2]

        # Step 3. Compute verification scores for each line in the eval list
        scores, labels = [], []
        for line in lines:
            label = int(line.split()[0])
            file1 = line.split()[1]
            file2 = line.split()[2]

            embedding_11, embedding_12 = embeddings[file1]
            embedding_21, embedding_22 = embeddings[file2]

            # Calculate verification scores 
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = ((score_1 + score_2) / 2).detach().cpu().numpy()

            scores.append(score)
            labels.append(label)

        # Step 4. Compute EER and minDCF using utility functions
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

        return EER, minDCF


    def save_parameters(self, path):
        """
        Save the model parameters to a file.

        Args:
            path (str): Target filepath for .pth or .pt model state.
        """
        torch.save(self.state_dict(), path)


    def load_parameters(self, path):
        """
        Load model parameters from a file.

        Args:
            path (str): Filepath to the saved model state (.pth or .pt).
        """
        self_state = self.state_dict()
        loaded_state = torch.load(path)

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                # Attempt to remove "module." prefix if present
                name = name.replace("module.", "")
                if name not in self_state:
                    print(f"{origname} is not recognized in the current model.")
                    continue

            # Check for mismatch in parameter shape
            if self_state[name].size() != loaded_state[origname].size():
                print(
                    "Mismatched parameter dimensions:"
                    f"{origname}, model: {self_state[name].size()}, "
                    f"loaded: {loaded_state[origname].size()}"
                )
                continue

            # Copy parameter
            self_state[name].copy_(param)

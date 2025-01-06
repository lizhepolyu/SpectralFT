import os
import glob
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from scipy import signal


def compute_db(waveform: np.ndarray) -> float:
    """
    Compute the dB value of an audio signal.
    
    Args:
        waveform (np.ndarray): Input one-dimensional audio data.

    Returns:
        float: The logarithmic decibel value of the audio's average power.
    """
    val = max(0.0, np.mean(np.power(waveform, 2)))
    return 10 * np.log10(val + 1e-4)


class TrainDataset(Dataset):
    """
    Audio dataset class for training, including data loading, trimming, padding, and augmentation.
    
    Attributes:
        train_list (str): Path to the text file containing "speaker ID" and "audio filename".
        train_path (str): Root directory where training audio files are stored.
        musan_path (str): Root directory of the MUSAN dataset (used for noise augmentation).
        rir_path (str): Directory where RIR (Room Impulse Response) files are stored.
        num_frames (int): Number of frames used during training.
        length (int): Target audio length (number of samples).
        data_list (List[str]): List of full paths to audio files.
        data_label (List[int]): List of corresponding speaker labels.
        noise_types (List[str]): List of noise types.
        noise_snr (dict): SNR ranges for each noise type.
        num_noise (dict): Range of the number of noises to mix for each noise type.
        noise_list (dict): Lists of available noise file paths for each noise type.
        rir_files (List[str]): List of RIR file paths.
    """

    def __init__(
        self,
        train_list: str,
        train_path: str,
        musan_path: str,
        rir_path: str,
        num_frames: int,
        **kwargs
    ):
        """
        Initialize the TrainDataset.
        
        Args:
            train_list (str): Path to the text file containing "speaker ID" and "audio filename".
            train_path (str): Root directory where training audio files are stored.
            musan_path (str): Root directory of the MUSAN dataset (used for noise augmentation).
            rir_path (str): Directory where RIR (Room Impulse Response) files are stored.
            num_frames (int): Number of frames used during training.
            **kwargs: Additional optional arguments.
        """
        super().__init__()

        self.train_path = train_path
        self.num_frames = num_frames

        # Calculate target length (in samples)
        self.length = self.num_frames * 160 + 240

        # Lists to store audio file paths and corresponding labels
        self.data_list = []
        self.data_label = []

        # 1) Create a dictionary for speaker IDs
        with open(train_list, 'r') as f:
            lines = f.readlines()
        speakers = [line.strip().split()[0] for line in lines]
        speaker_dict = {spk: idx for idx, spk in enumerate(sorted(set(speakers)))}

        # 2) Parse the train_list file and store full file paths and labels
        for line in lines:
            speaker_label = speaker_dict[line.strip().split()[0]]
            file_name = os.path.join(self.train_path, line.strip().split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

        # 3) Preload MUSAN noise file lists
        self.noise_types = ['noise', 'speech', 'music']
        self.noise_snr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noise_list = {nt: [] for nt in self.noise_types}

        # Collect all available noise files from musan_path
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            # The second-level directory determines the noise type
            category = os.path.basename(os.path.dirname(os.path.dirname(file)))
            if category in self.noise_list:
                self.noise_list[category].append(file)

        # 4) Collect RIR files
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data_list)

    def __getitem__(self, index: int):
        """
        Read and return the audio sample and its label at the specified index.
        
        Main steps:
            1. Read audio from disk.
            2. (Optional) Apply speed augmentation.
            3. Trim or pad to fixed length.
            4. (Optional) Apply further augmentations like reverberation, noise, or volume changes.
            5. Convert to PyTorch Tensor and retrieve the corresponding label.
        
        Args:
            index (int): Index of the sample.
        
        Returns:
            (audio_aug, label):
                - audio_aug (torch.FloatTensor): Augmented audio.
                - label (int): Speaker ID.
        """
        # 1) Read audio
        audio_path = self.data_list[index]
        audio, sr = sf.read(audio_path)
        audio = np.float32(audio)

        # 2) Trim or pad to fixed length
        audio = self._process_audio(audio)

        # 3) For HuBERT, apply data augmentation with 60% probability; no augmentation for WavLM
        if random.random() < 0.6:
            audio_aug = self._augment_audio(audio)
        else:
            audio_aug = audio

        # Convert to PyTorch tensor
        audio_aug = torch.FloatTensor(audio_aug)

        # Retrieve corresponding label
        label = self.data_label[index]

        return audio_aug, label

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Trim or pad the audio to the target length.
        
        Args:
            audio (np.ndarray): Input audio.
        
        Returns:
            np.ndarray: Audio with fixed length.
        """
        if len(audio) <= self.length:
            shortage = self.length - len(audio)
            audio = np.pad(audio, (0, shortage), mode='wrap')
        else:
            start_frame = random.randint(0, len(audio) - self.length)
            audio = audio[start_frame:start_frame + self.length]

        return audio

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation based on a random number:
            1 -> Add reverberation
            2 -> Add noise ('noise')
            3 -> Add noise ('speech')
            4 -> Add noise ('music')
            5 -> Add both 'speech' and 'music' noise
            6 -> Add Gaussian noise and then add reverberation
            7 -> Change volume and then add reverberation
        
        Args:
            audio (np.ndarray): Input audio.
        
        Returns:
            np.ndarray: Augmented audio.
        """
        aug_type = random.randint(1, 7)
        if aug_type == 1:
            return self._add_reverb(audio)
        elif aug_type in [2, 3, 4]:
            noise_type = self.noise_types[aug_type - 2]  # [2->0, 3->1, 4->2]
            return self._add_noise(audio, noise_type)
        elif aug_type == 5:
            # Add 'speech' noise first, then 'music' noise
            audio_aug = self._add_noise(audio, 'speech')
            return self._add_noise(audio_aug, 'music')
        elif aug_type == 6:
            # Add Gaussian noise then add reverberation
            audio_aug = self._add_gaussian_noise(audio)
            return audio_aug
        elif aug_type == 7:
            # Change volume then add reverberation
            audio_aug = self._change_volume(audio)
            return audio_aug
        else:
            return audio

    def _add_reverb(self, audio: np.ndarray) -> np.ndarray:
        """
        Add reverberation to the audio using RIR files.
        
        Args:
            audio (np.ndarray): Input clean audio.
        
        Returns:
            np.ndarray: Reverberated audio.
        """
        if not self.rir_files:
            # If no RIR files are available, return the original audio
            return audio

        rir_file = random.choice(self.rir_files)
        rir, sr = sf.read(rir_file)
        rir = np.float32(rir)

        # Normalize the RIR
        rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-8)

        # Convolve and truncate to fixed length
        audio_reverb = signal.convolve(audio, rir, mode='full')[:self.length]
        return audio_reverb

    def _add_noise(self, audio: np.ndarray, noise_type: str) -> np.ndarray:
        """
        Add noise (noise/speech/music) to the audio.
        
        Args:
            audio (np.ndarray): Input audio.
            noise_type (str): Type of noise ('noise', 'speech', 'music').
        
        Returns:
            np.ndarray: Noisy audio.
        """
        # Compute dB of clean audio
        clean_db = compute_db(audio)

        # Determine the number of noise files to mix
        num_noise_range = self.num_noise[noise_type]
        num_to_add = random.randint(num_noise_range[0], num_noise_range[1])

        if len(self.noise_list[noise_type]) == 0:
            # If no noise files are found, return the original audio
            return audio

        # Randomly select noise files
        noise_files = random.sample(
            self.noise_list[noise_type],
            k=min(num_to_add, len(self.noise_list[noise_type]))
        )

        noises = []
        for nf in noise_files:
            noise_audio, sr = sf.read(nf)
            noise_audio = np.float32(noise_audio)

            # Trim or pad noise to target length
            noise_audio = self._process_audio(noise_audio)

            # Compute dB of noise
            noise_db = compute_db(noise_audio)

            # Randomly select SNR
            snr = random.uniform(*self.noise_snr[noise_type])

            # Adjust noise amplitude to achieve desired SNR
            noise_audio *= np.sqrt(10 ** ((clean_db - noise_db - snr) / 10))
            noises.append(noise_audio)

        # Sum all noises
        if len(noises) > 0:
            noise_sum = np.sum(noises, axis=0)
            audio_noised = audio + noise_sum
        else:
            audio_noised = audio

        return audio_noised

    def _add_gaussian_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Add Gaussian white noise to the audio.
    
        Args:
            audio (np.ndarray): Input audio.
    
        Returns:
            np.ndarray: Noisy audio.
        """
        snr = random.uniform(10, 25)
        clean_db = compute_db(audio)

        noise = np.random.randn(len(audio)).astype(np.float32)
        noise_db = compute_db(noise)

        # Adjust noise amplitude based on SNR
        noise *= np.sqrt(10 ** ((clean_db - noise_db - snr) / 10))
        return audio + noise

    def _change_volume(self, audio: np.ndarray) -> np.ndarray:
        """
        Randomly change the volume of the audio.
    
        Args:
            audio (np.ndarray): Input audio.
    
        Returns:
            np.ndarray: Audio with changed volume.
        """
        volume_factor = random.uniform(0.8, 1.0005)
        return audio * volume_factor

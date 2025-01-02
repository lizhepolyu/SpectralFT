import os
import glob
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from scipy import signal


class TrainDataset(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        """
        Args:
            train_list (str): Path to the text file containing speaker ID and audio file name.
            train_path (str): Root directory for the training audio files.
            musan_path (str): Directory containing MUSAN dataset for noise augmentation.
            rir_path (str): Directory containing RIR (Room Impulse Response) files.
            num_frames (int): Number of frames to use for training.
            **kwargs: Other optional arguments.
        """
        self.train_path = train_path
        self.num_frames = num_frames
        # The target length of audio (in samples) for training, with some padding
        self.length = self.num_frames * 160 + 240

        # Lists to store audio file paths and speaker labels
        self.data_list = []
        self.data_label = []

        # Build a dictionary for speaker IDs
        with open(train_list, 'r') as f:
            lines = f.readlines()
        speakers = [line.strip().split()[0] for line in lines]
        speaker_dict = {speaker: idx for idx, speaker in enumerate(sorted(set(speakers)))}

        # Parse each line in the train_list
        for line in lines:
            speaker_label = speaker_dict[line.strip().split()[0]]
            file_name = os.path.join(self.train_path, line.strip().split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

        # Pre-load file lists for noise augmentation
        self.noise_types = ['noise', 'speech', 'music']
        self.noise_snr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noise_list = {noise_type: [] for noise_type in self.noise_types}

        # Gather all noise files from MUSAN dataset
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            # The second-level directory name determines noise category
            category = os.path.basename(os.path.dirname(os.path.dirname(file)))
            if category in self.noise_list:
                self.noise_list[category].append(file)

        # Gather all RIR files
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def __len__(self):
        """Return the total number of audio files."""
        return len(self.data_list)

    def __getitem__(self, index):
        """
        Get one audio sample and its corresponding label.

        Steps:
            1. Read audio from disk.
            2. Cut or pad audio to the target length.
            3. (Optional) Apply audio augmentation.
            4. Return the audio tensor and speaker label.
        """
        # Read audio file
        audio_path = self.data_list[index]
        audio, sr = sf.read(audio_path)
        audio = np.float32(audio)

        # Process audio to fixed length
        audio = self._process_audio(audio)

        # Data augmentation
        # Currently set to 0.0 probability, so no augmentation is performed
        if random.random() < 0.0:
            audio_aug = self._augment_audio(audio)
            audio_aug = torch.FloatTensor(audio_aug)
        else:
            audio_aug = torch.FloatTensor(audio)
        label = self.data_label[index]

        return audio_aug, label

    def _process_audio(self, audio):
        """
        Pad or cut audio to match the target length.

        Args:
            audio (np.array): Audio data in float32 format.
        Returns:
            np.array: Processed audio with the target length.
        """
        if len(audio) <= self.length:
            # If audio is shorter than the target length, pad it
            shortage = self.length - len(audio)
            audio = np.pad(audio, (0, shortage), mode='wrap')
        else:
            # Randomly select a start point to cut the audio
            start_frame = np.random.randint(0, len(audio) - self.length)
            audio = audio[start_frame:start_frame + self.length]

        return audio

    def _augment_audio(self, audio):
        """
        Randomly select one of the augmentation methods.

        Methods:
            1 -> Add reverberation
            2,3,4 -> Add noise of type 'noise', 'speech', or 'music'
            5 -> Add both 'speech' and 'music' noise
        """
        aug_type = random.randint(1, 5)
        if aug_type == 1:
            return self._add_reverb(audio)
        elif aug_type in [2, 3, 4]:
            noise_type = self.noise_types[aug_type - 2]
            return self._add_noise(audio, noise_type)
        elif aug_type == 5:
            audio_aug = self._add_noise(audio, 'speech')
            return self._add_noise(audio_aug, 'music')
        else:
            return audio

    def _add_reverb(self, audio):
        """
        Add reverberation effect by convolving the audio with a random RIR file.

        Args:
            audio (np.array): Clean audio data.
        Returns:
            np.array: Reverberant audio.
        """
        rir_file = random.choice(self.rir_files)
        rir, sr = sf.read(rir_file)
        rir = np.float32(rir)

        # Normalize RIR
        rir = rir / np.sqrt(np.sum(rir ** 2))

        # Convolution with RIR and keep the same length
        audio_aug = signal.convolve(audio, rir, mode='full')[:self.length]
        return audio_aug

    def _add_noise(self, audio, noise_type):
        """
        Add noise to the audio.

        Args:
            audio (np.array): Clean audio data.
            noise_type (str): Type of noise to add ('noise', 'speech', 'music').
        Returns:
            np.array: Audio with added noise.
        """
        # Compute clean audio dB
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        # Determine the number of noise files to be used
        num_noise = self.num_noise[noise_type]
        num = random.randint(num_noise[0], num_noise[1])

        # Randomly select noise files
        noise_files = random.sample(self.noise_list[noise_type], num)
        noises = []

        for noise_file in noise_files:
            noise_audio, sr = sf.read(noise_file)
            noise_audio = np.float32(noise_audio)

            # Process noise to match the target length
            noise_audio = self._process_audio(noise_audio)

            # Compute noise dB
            noise_db = 10 * np.log10(np.mean(noise_audio ** 2) + 1e-4)

            # Randomly choose SNR within the defined range
            snr = random.uniform(*self.noise_snr[noise_type])

            # Scale noise to match desired SNR
            noise_audio = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise_audio
            noises.append(noise_audio)

        # Sum up all noise signals
        noise = np.sum(noises, axis=0)

        # Add noise to the clean audio
        audio_aug = audio + noise
        return audio_aug
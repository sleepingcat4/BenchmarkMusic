import os
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioFolderDataset(Dataset):
    def __init__(self, root, sample_rate=16000, duration=2.0, n_fft=512, hop_length=256):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.files = []
        self.labels = []
        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            folder = os.path.join(root, c)
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)
        if waveform.shape[0] < self.num_samples:
            pad_len = self.num_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:self.num_samples]
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        log_magnitude = torch.log1p(magnitude)
        return log_magnitude, self.labels[idx]

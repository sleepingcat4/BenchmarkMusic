from imports import *

class AudioDatasetFromSingleFolder(Dataset):
    def __init__(self, root, sample_rate=16000, duration=2.0, n_fft=512, hop_length=256):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.files = []
        self.labels = []

        classes = ["ai", "human"]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            folder = os.path.join(root, c)
            print(f"Reading folder: {folder}")
            if not os.path.exists(folder):
                print(f"Folder {folder} does not exist, skipping")
                continue

            files_in_folder = [f for f in os.listdir(folder) if f.endswith(".mp3")]
            for f in tqdm(files_in_folder, desc=f"Processing '{c}'", ncols=80):
                self.files.append(os.path.join(folder, f))
                self.labels.append(self.class_to_idx[c])

            print(f"  Found {len(files_in_folder)} files in class '{c}'")

        print(f"Classes found: {len(self.class_to_idx)}")
        print(f"Data processing finished. Total samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.mean(dim=0)

        if waveform.shape[0] < self.num_samples:
            pad_len = self.num_samples - waveform.shape[0]
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:self.num_samples]

        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        log_magnitude = torch.log1p(magnitude)
        log_magnitude = (log_magnitude - log_magnitude.mean()) / (log_magnitude.std() + 1e-9)

        return log_magnitude, self.labels[idx]


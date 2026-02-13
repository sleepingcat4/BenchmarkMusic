from imports import *
from data import AudioDatasetFromSingleFolder
from model import FakeBenchmark
import os
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "."
batch_size = 64
epochs = 20
lr = 1e-3
n_fft = 512

num_workers = os.cpu_count()

dataset = AudioDatasetFromSingleFolder(root=root, n_fft=n_fft)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True
)

num_classes = 2
model = FakeBenchmark(num_classes=num_classes, n_fft=n_fft).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

torch.save(model.state_dict(), "detector.pth")


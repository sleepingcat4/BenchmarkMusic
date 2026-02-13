from imports import *
from data import AudioDatasetFromSingleFolder
from model import FakeBenchmark

device = "cuda" if torch.cuda.is_available() else "cpu"

eval_root = "./test"
eval_dataset = AudioDatasetFromSingleFolder(root=eval_root)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

model = FakeBenchmark(num_classes=2).to(device)
model.load_state_dict(torch.load("detector.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, y in eval_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        preds = outputs.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"Eval Accuracy: {accuracy * 100:.2f}%")


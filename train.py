import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm  # <--- Progress bar pro

# ========== PARAMETERS ==========
DATA_DIR = '../data'
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')

# ========== TRANSFORMATIONS ==========
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ========== DATASETS ==========
dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
CLASS_NAMES = dataset.classes
print(f"✅ Found classes: {CLASS_NAMES}")

# ========== MODEL ==========
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model = model.to(DEVICE)

# ========== OPTIMIZER & LOSS ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ========== TRAINING ==========
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc=f"🟦 Epoch {epoch+1}/{total_epochs}", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "Batch Acc": f"{(preds == labels).float().mean().item() * 100:.2f}%",
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(f"✅ [EPOCH {epoch+1}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


print("\n--- 🚀 STARTING TRAINING ---")
best_acc = 0.0
for epoch in range(EPOCHS):
    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS)
    if acc > best_acc:
        torch.save(model.state_dict(), "cv_finals.pth")
        best_acc = acc
        print(f"💾 Saved new model with acc={acc:.2f}% !")

print(f"✅ End of training. Best accuracy: {best_acc:.2f}%")
print("Saved final model cv_finals.pth")

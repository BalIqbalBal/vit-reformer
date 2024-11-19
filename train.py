import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from reformer.reformer_pytorch import ViRWithArcMargin
from utils.datasets import get_dataloaders
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard
writer = SummaryWriter("runs/vir_training")

# Create checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Dataset 
data_dir = "path/to/your/dataset"  # Root directory containing 'train' and 'test' subfolders
batch_size = 32
train_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=4)

# Define model
num_classes = len(train_loader.dataset.labels)
model = ViRWithArcMargin(
    image_size=224, 
    patch_size=16, 
    num_classes=num_classes, 
    dim=256, 
    depth=12, 
    heads=8, 
    arc_s=30.0, 
    arc_m=0.50
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training and evaluation function
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images, labels)  # Forward pass
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        train_loader_tqdm.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}")
    writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1", f1, epoch)

    return acc

def evaluate(epoch):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images, labels)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")
    writer.add_scalar("Test/Loss", total_loss / len(test_loader), epoch)
    writer.add_scalar("Test/Accuracy", acc, epoch)
    writer.add_scalar("Test/Precision", precision, epoch)
    writer.add_scalar("Test/Recall", recall, epoch)
    writer.add_scalar("Test/F1", f1, epoch)

    return acc

# Training loop
num_epochs = 50
best_accuracy = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_accuracy = train_one_epoch(epoch)
    test_accuracy = evaluate(epoch)

    # Save checkpoint if test accuracy improves
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        print(f"New best accuracy: {best_accuracy:.4f}. Model saved.")

# Close TensorBoard writer
writer.close()

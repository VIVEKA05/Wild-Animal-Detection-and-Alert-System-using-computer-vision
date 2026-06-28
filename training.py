import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = r"C:\Users\Vivek\Desktop\Animal_Dataset"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "validation")
TEST_DIR = os.path.join(DATASET_PATH, "test")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

device = torch.device("cpu")

print(f"\nUsing Device: {device}")

# ==========================================
# DATA AUGMENTATION
# ==========================================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transforms
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

CLASSES = train_dataset.classes

print("\nClasses Found:")
print(CLASSES)

NUM_CLASSES = len(CLASSES)

# ==========================================
# SAVE CLASS NAMES
# ==========================================

with open("class_names.txt", "w") as f:
    for cls in CLASSES:
        f.write(cls + "\n")

# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# LOAD PRETRAINED MODEL
# ==========================================

model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.DEFAULT
)

# Freeze backbone

for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(device)

print("\nModel Loaded Successfully")

# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

best_acc = 0.0

best_model_wts = copy.deepcopy(
    model.state_dict()
)

print("\nStarting Training...\n")

start_time = time.time()

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # -------------------
    # TRAIN
    # -------------------

    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        _, preds = torch.max(
            outputs,
            1
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * inputs.size(0)
        )

        running_corrects += torch.sum(
            preds == labels
        )

    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    epoch_acc = (
        running_corrects.double() /
        len(train_dataset)
    )

    # -------------------
    # VALIDATION
    # -------------------

    model.eval()

    val_corrects = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(
                outputs,
                1
            )

            val_corrects += torch.sum(
                preds == labels
            )

    val_acc = (
        val_corrects.double() /
        len(val_dataset)
    )

    print(
        f"Train Loss: {epoch_loss:.4f} | "
        f"Train Acc: {epoch_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        best_model_wts = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            model.state_dict(),
            "best_animal_model.pth"
        )

        print("Best model saved.")

# ==========================================
# LOAD BEST MODEL
# ==========================================

model.load_state_dict(
    best_model_wts
)

# ==========================================
# TEST ACCURACY
# ==========================================

model.eval()

test_correct = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(
            outputs,
            1
        )

        test_correct += torch.sum(
            preds == labels
        )

test_acc = (
    test_correct.double() /
    len(test_dataset)
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "final_animal_model.pth"
)

# ==========================================
# RESULTS
# ==========================================

elapsed = time.time() - start_time

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print(
    f"Best Validation Accuracy : {best_acc:.4f}"
)

print(
    f"Test Accuracy            : {test_acc:.4f}"
)

print(
    f"Training Time            : {elapsed/60:.2f} minutes"
)

print("\nSaved Files:")
print("best_animal_model.pth")
print("final_animal_model.pth")
print("class_names.txt")

print("\nClass Mapping:")

for i, cls in enumerate(CLASSES):
    print(f"{i} --> {cls}")
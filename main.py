from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# The model
class FractureDetector(nn.Module):
    def __init__(self):
        super(FractureDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(86528, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        
    def forward(self, x):
        x = self.maxpooling(self.relu(self.conv1(x)))
        x = self.maxpooling(self.relu(self.conv2(x)))
        x = self.maxpooling(self.relu(self.conv3(x)))
        x = self.classifier(x)

        return x

batch_size = 10
lr = 0.001
epochs = 5

# Data preperation
# transforms.Resize is from chatgpt
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train = datasets.ImageFolder("./Bone_Fracture_Binary_Classification/train", transform=transform)
test = datasets.ImageFolder("./Bone_Fracture_Binary_Classification/test", transform=transform)
validation = datasets.ImageFolder("./Bone_Fracture_Binary_Classification/val", transform=transform)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size)
validation_loader = DataLoader(validation, batch_size=batch_size)


model = FractureDetector().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
acc = Accuracy(task="binary").to(device)
test_acc = Accuracy(task="binary").to(device)
val_acc = Accuracy(task="binary").to(device)

def train():
    for epoch in range(epochs):
        total_loss = 0

        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            # Loss calculation is from chatgpt
            outputs = model(images).squeeze(1)
            loss = loss_fn(outputs, labels)
            acc.update(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        total_loss = total_loss / len(train_loader)
        print(f"Epoch Number {epoch + 1} | Loss {total_loss} | Accuracy {acc.compute()}")

        # Changing to evaluation mode
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            test_acc.update(outputs, labels)

        print("Test Accuracy:", test_acc.compute().item())
    
    # Changing to evaluation mode
    model.eval()
    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device).float()

        outputs = model(images).squeeze(1)
        val_acc.update(outputs, labels)

    print("Validation Accuracy:", val_acc.compute().item())

    print("Saving the model...")
    torch.save(model.state_dict(), "./FractureClassifier.pt")
    print("The model is saved.")

train()
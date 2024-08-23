import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from types import FunctionType
from torch import nn
from model import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


test_dir = "./data/test/"
train_dir = "./data/train/"
valid_dir = "./data/valid/"


device = (
    "cuda"
    if torch.cuda.is_available()

    else "mps"

    if torch.backends.mps.is_available()

    else "cpu"


)

MODEL = NeuralNetwork().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MODEL.parameters(), lr=1e-3)


print(f"Using device {device}")

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])


test_dataset = datasets.ImageFolder(test_dir, transform=transform)
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)


def load_and_shape(train: DataLoader, test: DataLoader, valid: DataLoader, batch_size: int = 10, shuffle: bool = False) -> DataLoader:

    train_dataloader = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, shuffle=shuffle)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W] {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader, valid_dataloader


train_loader, test_loader,  valid_loader = load_and_shape(
    train=train_dataset, test=test_dataset, valid=valid_dataset)


def train(dataloader: DataLoader, model, loss_fn: FunctionType, optimizer):

    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")


def test(dataloader: DataLoader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Store predictions and actual labels for plotting
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {
          (100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return all_preds, all_labels


def validate(dataloader: DataLoader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {
          (100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")


epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_loader, MODEL, loss_function, optimizer)
    validate(valid_loader, MODEL, loss_function)
    test(test_loader, MODEL, loss_function)
print("Done!")


all_preds, all_labels = test(test_loader, MODEL, loss_function)

# Plot Predicted vs Actual Labels
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Predicted vs Actual")
plt.show()

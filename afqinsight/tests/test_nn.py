import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def mlp4(input_shape, n_classes, output_activation="softmax", verbose=False):
    # Z. Wang, W. Yan, T. Oates, "Time Series Classification from Scratch with
    # Deep Neural Networks: A Strong Baseline," Int. Joint Conf.
    # Neural Networks, 2017, pp. 1578-1585

    ip = Input(shape=input_shape)

    fc = Flatten()(ip)

    fc = Dropout(0.1)(fc)
    fc = Dense(500, activation="relu")(fc)
    fc = Dropout(0.2)(fc)

    fc = Dense(500, activation="relu")(fc)
    fc = Dropout(0.2)(fc)

    fc = Dense(500, activation="relu")(fc)
    fc = Dropout(0.3)(fc)

    out = Dense(n_classes, activation=output_activation)(fc)

    model = Model([ip], [out])
    if verbose:
        model.summary()

    return model


dataset = torchvision.datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = mlp4(input_shape=784, n_classes=10).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train"):
        x, y = batch
        x = x.view(-1, 28 * 28).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for batch in tqdm(test_dataloader, desc=f"Epoch {epoch+1} Test"):
            x, y = batch
            x = x.view(-1, 784).to(device)
            y = y.to(device)
            yhat = model(x)
            test_loss += criterion(yhat, y).item()
            pred = yhat.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= len(test_dataloader.dataset)
        accuracy = 100.0 * correct / len(test_dataloader.dataset)
        print(f"Test loss: {test_loss}, Accuracy: {accuracy}%")

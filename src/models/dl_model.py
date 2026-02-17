import torch
import torch.nn as nn
import torch.optim as optim

class SimpleAudioNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleAudioNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DLModel:
    def __init__(self, input_dim, num_classes):
        self.model = SimpleAudioNet(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            inputs = torch.tensor(X_train, dtype=torch.float32)
            labels = torch.tensor(y_train, dtype=torch.long)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            return preds.numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

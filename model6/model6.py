import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # New hidden layer
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.LeakyReLU()  # Changed to LeakyReLU

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)  # Pass through new hidden layer
        out = self.fc3(out)
        return out


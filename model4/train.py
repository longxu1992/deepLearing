import torch
import torch.nn as nn
from data_reader import DataReader
from model4 import MLP
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(data_reader, model, batch_size, num_epochs):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        data_reader.offset = 0
        X, y = data_reader.get_batch(batch_size)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        while X is not None and y is not None:
            inputs = torch.from_numpy(X.astype(np.float32)).to(device)
            targets = torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每训练到一定阶段保存一下

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

            X, y = data_reader.get_batch(batch_size)


        # Save the model checkpoint
        torch.save(model.state_dict(), 'model.ckpt')


def main():
    input_size = 229
    hidden_size = 1024
    num_classes = 1
    batch_size = 1024  # You can adjust this number according to your requirement
    num_epochs = 10000  # You can adjust this number according to your requirement

    data_reader = DataReader('../东方财富0527.db', 'zjpc_num_final')
    model = MLP(input_size, hidden_size, num_classes)

    train_model(data_reader, model, batch_size, num_epochs)


if __name__ == '__main__':
    main()

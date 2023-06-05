import torch
from data_reader import DataReader
from model5 import MLP
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(data_reader, model, batch_size):
    model = model.to(device)
    X, y = data_reader.get_batch(batch_size)
    while X is not None and y is not None:
        inputs = torch.from_numpy(X.astype(np.float32)).to(device)
        targets = torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)

        # Forward pass
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs).data > 0.5).long()  # Convert to LongTensor
        targets = targets.long()  # Convert to LongTensor

        # Calculate accuracy
        accuracy = (predicted == targets).sum().item() / targets.numel()
        print(f'Accuracy: {accuracy}')

        # Calculate sensitivity
        true_positives = (predicted & targets).sum().item()
        actual_positives = targets.sum().item()
        sensitivity = true_positives / actual_positives if actual_positives > 0 else 0
        print(f'Sensitivity: {sensitivity}')

        X, y = data_reader.get_batch(batch_size)


def main():
    input_size = 229
    hidden_size = 1024
    num_classes = 1
    batch_size = 100000  # You can adjust this number according to your requirement

    data_reader = DataReader('../东方财富0527.db', 'zjpc_num_final')
    model = MLP(input_size, hidden_size, num_classes)

    # Load model
    model.load_state_dict(torch.load('model_epoch_8300.pth'))

    test_model(data_reader, model, batch_size)


if __name__ == '__main__':
    main()
# Accuracy: 0.9602555003548616
# Sensitivity: 0.975899875879189

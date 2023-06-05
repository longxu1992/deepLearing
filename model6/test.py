import torch
from data_reader import DataReader
from model6 import MLP
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
        predicted = torch.sigmoid(outputs).data > 0.5
        accuracy = (predicted == targets).sum().item() / targets.numel()

        print(f'Accuracy: {accuracy}')

        X, y = data_reader.get_batch(batch_size)


def main():
    input_size = 229
    hidden_size = 1024
    num_classes = 1
    batch_size = 100  # You can adjust this number according to your requirement

    data_reader = DataReader('../东方财富0527.db', 'zjpc_num_final')
    model = MLP(input_size, hidden_size, num_classes)

    # Load model
    # model.load_state_dict(torch.load('model.ckpt'))
    model.load_state_dict(torch.load('model_epoch_300.pth'))

    test_model(data_reader, model, batch_size)


if __name__ == '__main__':
    main()

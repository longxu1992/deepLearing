import torch
from datareader.DataModelEnum import DataModelEnum
from datareader.data_reader_prd import DataReaderPrd
from model7 import MLP
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_model(model, input_data):
    model = model.to(device)
    inputs = input_data.to(device).float()

    # Forward pass
    outputs = model(inputs)
    predicted = (torch.sigmoid(outputs).data > 0.5).long()  # Convert to LongTensor

    return predicted.cpu().numpy()


def main():
    data_base = DataModelEnum.M7PRD
    model_base = DataModelEnum.M7D3
    input_size = data_base.input_size
    hidden_size = 1024
    num_classes = 1
    data_reader = DataReaderPrd(data_base, 24, 5)
    model = MLP(input_size, hidden_size, num_classes)

    # Load model
    model.load_state_dict(torch.load(f'{model_base.name}model_epoch_900.pth'))
    X = data_reader.get_batch()
    # Here you should load your actual data
    input_data = torch.from_numpy(X.astype(np.float32)).to(device)
    result = []

    predictions = predict_model(model, input_data)
    # Print input data and corresponding predictions
    for i in range(len(input_data)):
        input_info = input_data[i].cpu().numpy().tolist()
        if predictions[i] == 1:
            print('Input:', input_info)
            result.append(input_info[-1])
            print('Prediction:', 'Positive' if predictions[i] == 1 else 'Negative')
            print()
    print(result)


if __name__ == '__main__':
    main()

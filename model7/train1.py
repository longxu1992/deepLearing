from datareader.DataModelEnum import DataModelEnum
from datareader.data_reader import DataReader
from model7 import MLP
from train_model import train_model


def main():
    data_base = DataModelEnum.M7D1
    input_size = data_base.input_size
    hidden_size = 1024
    num_classes = 1
    batch_size = 1024  # You can adjust this number according to your requirement
    num_epochs = 10000  # You can adjust this number according to your requirement

    data_reader = DataReader(data_base)
    model = MLP(input_size, hidden_size, num_classes)

    train_model(data_reader, model, batch_size, num_epochs)


if __name__ == '__main__':
    main()

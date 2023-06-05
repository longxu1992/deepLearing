import torch
import torch.nn as nn

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(data_reader, model, batch_size, num_epochs):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    for epoch in range(num_epochs):
        data_reader.offset = 0
        X, y = data_reader.get_batch(batch_size)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'{data_reader.model_db}model_epoch_{epoch + 1}.pth')
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
        torch.save(model.state_dict(), f'{data_reader.model_db}model.ckpt')

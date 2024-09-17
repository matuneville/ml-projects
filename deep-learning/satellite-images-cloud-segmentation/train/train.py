import time
import torch
from .utils import *

def train(model, train_dl, valid_dl, loss_fn, optimizer, epochs, device):
    start_time = time.time()

    model.to(device)

    train_hist_acc = [0.] * epochs
    train_hist_loss = [0.] * epochs
    valid_hist_acc = [0.] * epochs
    valid_hist_loss = [0.] * epochs

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        train_correct = 0

        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch).squeeze()  # 1. Forward pass
            loss = loss_fn(pred, y_batch.float())  # 2. Calc loss

            optimizer.zero_grad()  # 3. Restart gradients
            loss.backward()  # 4. Backward pass
            optimizer.step()  # 5. Step forward

            train_loss += loss.item() * x_batch.size(0)  # Accumulate loss
            train_correct += compute_accuracy(pred, y_batch) * x_batch.size(0)  # Accumulate correct predictions

        model.eval()
        valid_loss = 0.0
        valid_correct = 0

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch).squeeze()  # 1. Predict
                loss = loss_fn(pred, y_batch.float())  # 2. Calc loss

                valid_loss += loss.item() * x_batch.size(0)  # Accumulate loss
                valid_correct += compute_accuracy(pred, y_batch) * x_batch.size(0)  # Accumulate correct predictions

        train_hist_loss[epoch] = train_loss / len(train_dl.dataset)
        train_hist_acc[epoch] = train_correct / len(train_dl.dataset)
        valid_hist_loss[epoch] = valid_loss / len(train_dl.dataset)
        valid_hist_acc[epoch] = valid_correct / len(train_dl.dataset)

        torch.cuda.empty_cache()

        elapsed_time = (time.time() - start_time) / 60
        print('--------------------------------------------------')
        print(f'Epoch {epoch + 1}, time elapsed: {elapsed_time:.2f} min\n'
              f'Training accuracy: {train_hist_acc[epoch]:.4f}, Training loss: {train_hist_loss[epoch]:.4f}\n'
              f'Validation accuracy: {valid_hist_acc[epoch]:.4f}, Validation loss: {valid_hist_loss[epoch]:.4f}')

    return train_hist_acc, train_hist_loss, valid_hist_acc, valid_hist_loss

from .utils import *

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    train_loss, valid_loss = [], []
    train_acc_history, valid_acc_history = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs} -------------------------')

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                dataloader = train_dl
            else:
                model.train(False)
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0
            step = 0

            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y.long())
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                if step % 50 == 0:
                    print(f'Current step: {step}  Loss: {loss:.4f}  Acc: {acc:.4f}')

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
            train_acc_history.append(epoch_acc) if phase == 'train' else valid_acc_history.append(epoch_acc)

    return train_loss, valid_loss, train_acc_history, valid_acc_history
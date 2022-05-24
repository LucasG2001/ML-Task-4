import torch
from torchvision import models
import torch.optim as optim
import numpy as np
import time
from evaluate import accuracy, evaluate, plot

def train_network_complete(model, dataloader, val_loader, optim, device, criterion=torch.nn.MSELoss(), number_epochs=30, mode='encoder'):
    save_epoch = 10
    train_acc_arr = np.zeros(number_epochs)
    val_acc_arr = np.zeros(number_epochs)

    for epoch in range(number_epochs):
        # reset statistics trackers
        train_loss_cum = 0.0
        train_acc_cum = 0.0
        num_samples_epoch = 0
        t = time.time()

        for batch in dataloader:
            # extract images from train loader
            features, labels = batch
            num_samples_batch = len(batch[0])

            features = features.float().to(device)
            labels = labels.float().to(device)
            # zero grads and put model into train mode
            optim.zero_grad(set_to_none=True)
            model.train()

            ## forward pass
            output = model(features)

            ##compute loss
            if mode == 'encoder':
                #loss = criterion(output, features, torch.ones(num_samples_batch, device=device))  # Autoencoder: Target = features = input, cosine loss
                loss = criterion(output, features)  # Autoencoder: Target = features = input
            elif mode =='net':
                loss = criterion(output, labels)  # Autoencoder: Target = features = input

            ## backward pass and gradient step
            loss.backward()
            optim.step()

            # keep track of train stats

            num_samples_epoch += num_samples_batch

            train_loss_cum += loss * num_samples_batch

        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_train_acc = evaluate(model, dataloader, mode=mode)
        avg_val_acc = evaluate(model, val_loader, mode=mode)
        epoch_duration = time.time() - t

        train_acc_arr[epoch] = avg_train_acc
        val_acc_arr[epoch] = avg_val_acc

        # print some infos
        print(
            f'Epoch {epoch} | Train loss: {train_loss_cum.item():.4f} | Train accuracy: {avg_train_acc:.4f} | Validation accuracy: {avg_val_acc:.4f} |'
            f' Duration {epoch_duration:.2f} sec')

        # save checkpoint of model
        if epoch % save_epoch == 0 and epoch > 0:
            save_path = f'./model_instances/model_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict()},
                       save_path)
            print(f'Saved model checkpoint to {save_path}')

    save_path = f'./model_instances/model_final.pt'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()},
               save_path)
    print(f'Saved final model to {save_path}')

    return model, number_epochs, train_acc_arr, val_acc_arr

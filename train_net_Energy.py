import numpy as np
import pandas as pd
import torch
from dataset import LumoEnergiesData
from network_classes import Autoencoder, EnergyPredictor
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from training_loop_autoencoder import train_network_complete
import torch.nn as nn
from evaluate import accuracy, evaluate, plot
import time
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device ", device)
train_size = 37500
val_size = 12500

encoded_features_df = pd.read_csv("encoded_features.csv")
labels_df = pd.read_csv("./task4Data/CSV_Data/pretrain_labels.csv")
encoded_features_df.drop(columns=encoded_features_df.columns[0], axis=1, inplace=True)
labels_df.drop(['Id'], axis='columns', inplace=True)

encoded_features = encoded_features_df.to_numpy()
labels = labels_df.to_numpy()

scaler = StandardScaler()
encoded_features = scaler.fit_transform(encoded_features)

print("Creating Datasets & Dataloaders")
LumoEnergy = LumoEnergiesData(features=encoded_features, labels=labels)
used_datapoints = list(range(0, train_size + val_size))  # Define used indices of complete dataset
dataset_reduced = Subset(LumoEnergy, used_datapoints)
[train_data, val_data] = random_split(dataset_reduced, [train_size, val_size])

dataloader = DataLoader(dataset=train_data,batch_size=256,shuffle=True,num_workers=8,pin_memory=True)
val_loader = DataLoader(dataset=val_data,batch_size=256,shuffle=True,num_workers=8,pin_memory=True)


energy_predictor = EnergyPredictor()
energy_predictor = energy_predictor.to(device)

number_epochs = 100
save_epoch = 10
optim = optim.SGD(energy_predictor.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
criterion = nn.MSELoss()
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

        features = features.float().to(device)
        labels = labels.float().to(device)
        # zero grads and put model into train mode
        optim.zero_grad(set_to_none=True)
        energy_predictor.train()

        ## forward pass
        predicted_energies = energy_predictor(features)

        ##compute loss
        loss = criterion(predicted_energies, labels)  # Net: Target = labels

        ## backward pass and gradient step
        loss.backward()
        optim.step()


        # keep track of train stats
        num_samples_batch = len(batch[0])
        num_samples_epoch += num_samples_batch

        train_loss_cum += loss * num_samples_batch

    # average the accumulated statistics
    avg_train_loss = train_loss_cum / num_samples_epoch
    avg_train_acc = evaluate(energy_predictor, dataloader, mode='net')
    avg_val_acc = evaluate(energy_predictor, val_loader, mode='net')
    epoch_duration = time.time() - t

    train_acc_arr[epoch] = avg_train_acc
    val_acc_arr[epoch] = avg_val_acc

    # print some infos
    print(
        f'Epoch {epoch} | Train loss: {train_loss_cum.item():.4f} | Train accuracy: {avg_train_acc:.4f} | Validation accuracy: {avg_val_acc:.4f} |'
        f' Duration {epoch_duration:.2f} sec')

    # save checkpoint of model
    if epoch % save_epoch == 0 and epoch > 0:
        save_path = f'./model_instances/EnergyPredictorModel_epoch_{epoch}.pt'
        torch.save({'epoch': epoch,
                    'model_state_dict': energy_predictor.state_dict(),
                    'optimizer_state_dict': optim.state_dict()},
                   save_path)
        print(f'Saved model checkpoint to {save_path}')

save_path = f'./model_instances/EnergyPredictorModel_final.pt'
torch.save({'epoch': epoch,
            'model_state_dict': energy_predictor.state_dict(),
            'optimizer_state_dict': optim.state_dict()},
           save_path)
print(f'Saved final model to {save_path}')

plot(number_epochs, train_acc_arr, val_acc_arr, name='accuracy_plot_net')
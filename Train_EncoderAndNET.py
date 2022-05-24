import numpy as np
import pandas as pd
import torch
from dataset import LumoEnergiesData
from network_classes import EnergyNet, FeatureExtractor
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from training_loop import train_network_complete
import torch.nn as nn
from evaluate import plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

torch.manual_seed(123)
np.random.seed(123)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device ", device)

pretrain_features_df = pd.read_csv("./task4Data/CSV_Data/pretrain_features.csv")
pretrain_labels_df = pd.read_csv("./task4Data/CSV_Data/pretrain_labels.csv")
pretrain_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
pretrain_labels_df.drop(['Id'], axis='columns', inplace=True)

pretrain_features = pretrain_features_df.to_numpy()
pretrain_labels = pretrain_labels_df.to_numpy()

cutoff_low = (-4.4)
cutoff_high = (-2.1)

pretrain_labels_high = pretrain_labels[np.where(pretrain_labels[:, 0] > cutoff_high)]
pretrain_labels_low = pretrain_labels[np.where(pretrain_labels[:, 0] < cutoff_low)]
pretrain_features_high = pretrain_features[np.where(pretrain_labels[:, 0] > cutoff_high)]
pretrain_features_low = pretrain_features[np.where(pretrain_labels[:, 0] < cutoff_low)]

for i in range(50):
    pretrain_labels = np.concatenate((pretrain_labels, pretrain_labels_high), axis=0)
    pretrain_features = np.concatenate((pretrain_features, pretrain_features_high), axis=0)

for j in range(50):
    pretrain_labels = np.concatenate((pretrain_labels, pretrain_labels_low), axis=0)
    pretrain_features = np.concatenate((pretrain_features, pretrain_features_low), axis=0)


plt.hist(pretrain_labels, bins=50)
plt.xlabel('value')
plt.ylabel('occurrence')
plt.title('pretrain_labels')
plt.savefig('pretrain_labels_added.png')
plt.close()


scaler = StandardScaler()
pretrain_labels = scaler.fit_transform(pretrain_labels)

train_size = 37500
val_size = 12500

print("Creating Datasets & Dataloaders")
LumoEnergy = LumoEnergiesData(features=pretrain_features, labels=pretrain_labels)
used_datapoints = list(range(0, train_size + val_size))  # Define used indices of complete dataset
dataset_reduced = Subset(LumoEnergy, used_datapoints)
[train_data, val_data] = random_split(dataset_reduced, [train_size, val_size])

complete_dataloader = DataLoader(dataset=LumoEnergy,batch_size=1024,shuffle=False,num_workers=8,pin_memory=True)
dataloader = DataLoader(dataset=train_data,batch_size=1024,shuffle=True,num_workers=8,pin_memory=True)
val_loader = DataLoader(dataset=val_data,batch_size=1024,shuffle=False,num_workers=8,pin_memory=True)

print("Generating Model")
net = FeatureExtractor()
net = net.to(device)

print("Defining loss and optimizer")
optim = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
adam = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.007)#best: 0.001, 0.007
mseloss = nn.MSELoss()
cosLoss = nn.CosineEmbeddingLoss()

print("Starting Training")
net, n_epochs, train_acc, val_acc = train_network_complete(model=net, dataloader=dataloader, val_loader=val_loader,
                                                           optim=adam, device=device, criterion=mseloss,
                                                           number_epochs = 20, mode='net')
#note "undertraining" will lead to high bias, no variance and false predicitons between 0 and -2 instead of 0 and -4
# also high batchsizes >= 512 will lead to bad predictions with no variance
plot(n_epochs, train_acc, val_acc,'accuracy_plot_encoder')

# Make Predictions
net.eval()
lumo_predictions_torch = torch.empty(0, device=device)
lumo_energies = torch.empty(0, device=device)
x = np.zeros((val_size, 1))
print("making predictions")
for i, data in enumerate(val_loader):
    features, labels = data
    features = features.float().to(device)
    labels = labels.float().to(device)

    lumo_energy_predicted = net(features)

    lumo_predictions_torch = torch.cat((lumo_predictions_torch, lumo_energy_predicted), dim=0)
    lumo_energies = torch.cat((lumo_energies, labels), dim=0)
    print(i)


e_predictions = lumo_predictions_torch.detach().cpu().numpy()
lumo_energies = lumo_energies.detach().cpu().numpy()
x_axis = np.arange(val_size)
plt.plot(x_axis, lumo_energies, label='real')
plt.plot(x_axis, e_predictions, label='Predicted')
plt.legend()
plt.grid()
plt.savefig('E_lumo_prediction_small_B1024_E32.png')
e_predictions_df = pd.DataFrame(e_predictions)
e_predictions_df.to_csv('lumo_energy_predictions.csv')
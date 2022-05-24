import numpy as np
import pandas as pd
import torch
from dataset import LumoEnergiesData
from network_classes import EnergyNet, FeatureExtractor
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from training_loop import train_network_complete
import torch.nn as nn
from evaluate import plot, evaluate, accuracy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.manual_seed(123)
np.random.seed(123)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device ", device)

train_features_df = pd.read_csv("./task4Data/CSV_Data/train_features.csv")
train_labels_df = pd.read_csv("./task4Data/CSV_Data/train_labels.csv")
test_features_df = pd.read_csv("./task4Data/CSV_Data/test_features.csv")
test_features_id = test_features_df.iloc[:, 0]

test_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
train_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
train_labels_df.drop(['Id'], axis='columns', inplace=True)

train_features = train_features_df.to_numpy()
train_labels = train_labels_df.to_numpy()
test_features = test_features_df.to_numpy()

cutoff_high = 2.4
cutoff_low = 1.4

train_labels_high = train_labels[np.where(train_labels[:, 0] > cutoff_high)]
train_labels_low = train_labels[np.where(train_labels[:, 0] < cutoff_low)]
train_features_high = train_features[np.where(train_labels[:, 0] > cutoff_high)]
train_features_low = train_features[np.where(train_labels[:, 0] < cutoff_low)]

for i in range(2):
    train_labels = np.concatenate((train_labels, train_labels_high), axis=0)
    train_features = np.concatenate((train_features, train_features_high), axis=0)

for j in range(2):
    train_labels = np.concatenate((train_labels, train_labels_low), axis=0)
    train_features = np.concatenate((train_features, train_features_low), axis=0)

plt.hist(train_labels, bins=50)
plt.xlabel('value')
plt.ylabel('occurrence')
plt.title('train_labels')
plt.savefig('train_labels_added.png')


feature_extract = True
model_loaded = FeatureExtractor()
checkpoint = torch.load('./model_instances/SmallNet1024.pt')  # adjust manually for best epoch
model_loaded.load_state_dict(checkpoint['model_state_dict'])  # load saved parameters
set_parameter_requires_grad(model_loaded, feature_extracting=False)
model_loaded.fc = nn.Sequential(nn.Linear(100, 50), nn.LeakyReLU(), nn.Linear(50, 1)) # 100 auf 3 und 3 auf 1 bis jetzt am besten, batch = 40

model_loaded = model_loaded.to(device)  # load model

if feature_extract:
    params_to_update = []
    for name, param in model_loaded.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_loaded.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

print("Creating Datasets & Dataloaders")
GapData = LumoEnergiesData(features=train_features, labels=train_labels)
TestData = LumoEnergiesData(features=test_features, labels=np.zeros((10000, 1)))

dataloader = DataLoader(dataset=GapData, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(dataset=GapData, batch_size=24, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=TestData, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)
print("size of test data: ", len(test_loader))

print("Defining loss and optimizer")
optim = optim.SGD(params_to_update, lr=1e-2, momentum=0.9, weight_decay=1e-3)
adam = torch.optim.Adam(params_to_update, lr=0.0001, weight_decay=0.09) #best: lr = 0.0025, w_d = 0.09
mseloss = nn.MSELoss()

model_loaded, n_epochs, train_acc, val_acc = train_network_complete(model=model_loaded, dataloader=dataloader,
                                                                    val_loader=val_loader,
                                                                    optim=adam, device=device, criterion=mseloss,
                                                                    number_epochs=65, mode='net')
plot(n_epochs, train_acc, val_acc, 'FinetuneAccuracy.png')

# Make Predictions
model_loaded.eval()
gap_predictions_torch = torch.empty(0, device=device)

print("making predictions")
for i, data in enumerate(test_loader):
    encoded_features, labels = data
    encoded_features = encoded_features.float().to(device)

    predictions = model_loaded(encoded_features)

    gap_predictions_torch = torch.cat((gap_predictions_torch, predictions), dim=0)

    print(i)

gap_predictions = gap_predictions_torch.detach().cpu().numpy()
gap_predictions_df = pd.DataFrame(gap_predictions)
gap_predictions_df = pd.concat([test_features_id, gap_predictions_df], axis=1)
gap_predictions_df.columns = ['Id', 'y']
gap_predictions_df.to_csv('gap_predictions.csv')

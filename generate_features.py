import numpy as np
import pandas as pd
import torch
from dataset import LumoEnergiesData
from network_classes import Autoencoder
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from training_loop_autoencoder import train_network_complete
import torch.nn as nn
from evaluate import plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device ", device)

model = Autoencoder()
model_loaded = model.to(device)  # load model
checkpoint = torch.load('./model_instances/autoencoder_scaled.pt')  # adjust manually for best epoch
model_loaded.load_state_dict(checkpoint['model_state_dict'])  # load saved parameters


pretrain_features_df = pd.read_csv("./task4Data/CSV_Data/pretrain_features.csv")
pretrain_labels_df = pd.read_csv("./task4Data/CSV_Data/pretrain_labels.csv")
test_features_df = pd.read_csv("./task4Data/CSV_Data/test_features.csv")
SMALL_train_features_df = pd.read_csv("./task4Data/CSV_Data/train_features.csv")
SMALL_train_labels_df = pd.read_csv("./task4Data/CSV_Data/train_labels.csv")

test_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
pretrain_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
pretrain_labels_df.drop(['Id'], axis='columns', inplace=True)

SMALL_train_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
SMALL_train_labels_df.drop(['Id'], axis='columns', inplace=True)

test_features = test_features_df.to_numpy()
pretrain_features = pretrain_features_df.to_numpy()
pretrain_labels = pretrain_labels_df.to_numpy()
small_train_features = SMALL_train_features_df.to_numpy()
small_train_labels = SMALL_train_labels_df.to_numpy()


n, m = pretrain_features.shape  # should be  50'000 x 1'000
output_size = 50 # output size of autoencoder
reduced_features = np.zeros((50000,output_size))
reduced_TEST_features = np.zeros((10000,50))
reduced_small_features = np.zeros((100,50))
LumoEnergy = LumoEnergiesData(features=pretrain_features, labels=pretrain_labels)
TestData = LumoEnergiesData(features=test_features, labels=np.zeros((10000,1)))
SmallData = LumoEnergiesData(features=small_train_features, labels=small_train_labels)

dataloader = DataLoader(dataset=LumoEnergy,batch_size=512,shuffle=False,num_workers=8,pin_memory=True)
test_dataloader = DataLoader(dataset=TestData,batch_size=512,shuffle=False,num_workers=8,pin_memory=True)
small_dataloader = DataLoader(dataset=SmallData,batch_size=512,shuffle=False,num_workers=8,pin_memory=True)
print("length of test data: ", len(test_dataloader))

model.eval()
red_features_torch = torch.empty(0, device=device)
red_small_features_torch = torch.empty(0, device=device)
reduced_features_test_torch = torch.empty(0, device=device)

for i, (data) in enumerate(small_dataloader):
    inputs, labels = data
    inputs = inputs.float().to(device)
    reduced_input = model.encoder(inputs)
    red_small_features_torch = torch.cat((red_small_features_torch, reduced_input), dim=0)
    print(i)

reduced_small_features = red_small_features_torch.detach().cpu().numpy()
reduced_small_features_df = pd.DataFrame(reduced_small_features)
reduced_small_features_df.to_csv('encoded_small_features.csv')

for i, (data) in enumerate(dataloader):
    inputs, labels = data
    inputs = inputs.float().to(device)
    reduced_input = model.encoder(inputs)
    red_features_torch = torch.cat((red_features_torch, reduced_input), dim=0)
    print(i)

reduced_features = red_features_torch.detach().cpu().numpy()
reduced_features_df = pd.DataFrame(reduced_features)
reduced_features_df.to_csv('encoded_features.csv')

for i, test_data in enumerate(test_dataloader):
    test_inputs, test_labels = test_data
    test_inputs = test_inputs.float().to(device)
    reduced_test_features = model.encoder(test_inputs)
    reduced_features_test_torch = torch.cat((reduced_features_test_torch, reduced_test_features), dim=0)
    print(i)

reduced_TEST_features = reduced_features_test_torch.detach().cpu().numpy()
reduced_TEST_features_df = pd.DataFrame(reduced_TEST_features)
reduced_TEST_features_df.to_csv('encoded_Test_features.csv')


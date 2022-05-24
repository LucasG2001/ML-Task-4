import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pretrain_features_df = pd.read_csv("./task4Data/CSV_Data/pretrain_features.csv")
pretrain_labels = pd.read_csv("./task4Data/CSV_Data/pretrain_labels.csv")
train_features = pd.read_csv("./task4Data/CSV_Data/train_features.csv")
train_labels = pd.read_csv("./task4Data/CSV_Data/train_labels.csv")
test_features = pd.read_csv("./task4Data/CSV_Data/test_features.csv")

pretrain_features_df.drop(['Id', 'smiles'], axis='columns', inplace=True)
pretrain_labels.drop(['Id'], axis='columns', inplace=True)
train_features.drop(['Id', 'smiles'], axis='columns', inplace=True)
train_labels.drop(['Id'], axis='columns', inplace=True)
test_features.drop(['Id', 'smiles'], axis='columns', inplace=True)


plt.hist(train_labels, bins=50)
plt.xlabel('value')
plt.ylabel('occurrence')
plt.title('train_labels')
plt.savefig('train_labels.png')
plt.close()

plt.hist(pretrain_labels, bins=50)
plt.xlabel('value')
plt.ylabel('occurrence')
plt.title('PREtrain_labels')
plt.savefig('PREtrain_labels.png')
plt.close()



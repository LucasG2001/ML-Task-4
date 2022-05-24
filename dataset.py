class LumoEnergiesData:
    def __init__(self, features=None, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):

        feature = self.features[int(index)]
        label = self.labels[int(index)]

        if self.transform:
            feature = self.transform(feature)
            label = self.transform(label)

        return feature, label

    def __len__(self):
        return len(self.labels)
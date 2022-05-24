import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# compute prediction accuracy given the NN outputs
def accuracy(net_out, original_feature):
    diff = torch.sub(net_out, original_feature, alpha=1)
    return torch.norm(diff, p=2)


# evaluate trained model on validation data
def evaluate(ev_model: torch.nn.Module, ev_dataloader, mode='encoder') -> torch.Tensor:
    # goes through the test dataset and computes the test accuracy
    ev_model.eval()  # bring the model into eval mode
    with torch.no_grad():
        acc_cum = 0.0
        num_eval_samples = 0
        for batch in ev_dataloader:
            features, labels = batch
            features = features.float().to(device)
            labels = labels.float().to(device)

            num_samples_batch = len(batch[0])
            num_eval_samples += num_samples_batch
            out = ev_model(features)
            if mode == 'encoder':
                acc_cum += accuracy(out, features)
            elif mode == 'net':
                acc_cum += accuracy(out, labels)

        avg_acc = acc_cum / num_eval_samples
        return avg_acc


def plot(number_epochs, train_acc, val_acc, name):
    x_axis = np.arange(number_epochs)
    plt.plot(x_axis, train_acc, label='Training accuracy')
    plt.plot(x_axis, val_acc, label='Validation accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(name)
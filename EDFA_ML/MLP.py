from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
import argparse
from torch.utils.benchmark import Timer
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(85, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        ) 
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(128, 83)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.decoder(x)
        return y
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def __len__(self):
        return len(self.X)

start_time = time.time()
dataset = np.loadtxt("C:/Users/s1280/PycharmProjects/pythonProject1/EDFA_ML/DATASET/EDFA_1_15dB.csv", delimiter=",")
X = dataset[:, 1:86]
y = dataset[:, 86:]
epoch_loss = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN().to(device)

def train(model, train_loader, optimizer):
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # データをGPUに移動
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sqrt(nn.MSELoss()(output, target) + 1e-6)
        loss.backward()
        optimizer.step()
    scheduler.step()

def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)  # データをGPUに移動
            output = model(data)
            val_loss += torch.sqrt(nn.MSELoss()(output, target) + 1e-6).item()
    return val_loss / len(val_loader)

model = NN()
print(model)



def inference_best_model(model, dataloader, best_model_path):
    model.eval()

    total_flops = 0

    # ベストモデルの重みをロード
    best_model_state = torch.load(best_model_path)
    model.load_state_dict(best_model_state)

    with torch.no_grad():
        for data, _ in dataloader:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                _ = model(data)

            total_flops += prof.total_average().float_ops

    total_gflops = total_flops / 1e9  # FLOPをGFLOPに変換
    print(f"Total GFLOPs during inference with the best model: {total_gflops}")


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)

scheduler = CosineAnnealingLR(optimizer, 20, 1e-5)

loss_fn = nn.MSELoss()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
ls = []
params = model.state_dict()
print("Weights of layer1:", params['layer1.0.weight'])

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    train_dataset = MyDataset(X[train_index], y[train_index])
    val_dataset = MyDataset(X[val_index], y[val_index])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    for epoch in range(500):
        train(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        epoch_loss.append(val_loss)  # Store the validation loss for the epoch
        print("Fold {}, Epoch {}, Val Loss: {:.4f}".format(fold, epoch+1, val_loss))
        if fold == 0 and epoch == 0:
            best_loss = val_loss
        elif val_loss < best_loss:
            best_loss = val_loss
            filename = 'model_best_{}-{}.pth'.format(fold, epoch+1)
            torch.save(model.state_dict(), filename)
            print("********************************BEST {}-RMSE: {:.2e}*****************************".format(fold, best_loss))
        if epoch == 499:
            ls.append(best_loss)

# Save the loss values to a CSV file
p = pd.DataFrame(ls)
filename = 'loss.csv'
p.to_csv(filename)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"処理にかかった時間: {elapsed_time}秒")
# After the training loop, plot the loss values
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(list(range(len(epoch_loss))), epoch_loss)
ax.set_xlabel('#epoch')
ax.set_ylabel('MSE')
y_ticks = np.arange(0, max(epoch_loss), 0.4)  # 0から最大値まで0.01刻みの目盛
ax.set_yticks(y_ticks)
# Save the figure
fig.savefig('training_loss_plot_simple.png')


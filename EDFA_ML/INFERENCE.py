import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import profiler
import numpy as np
import os
from thop import profile
# モデルの定義
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


# 推論関数の定義
def inference_best_model(model, dataloader, best_model_path):
    state_dict = torch.load(best_model_path)

    # 不要なキーを削除
    #state_dict.pop("total_ops", None)
    #state_dict.pop("total_params", None)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    correct_predictions = 0
    total_samples = 0
    allowed_error = 0.1  # 10%以内の誤差を許容

    all_outputs = []   # すべてのバッチのモデルの出力を格納するリスト
    all_targets = []    # すべてのバッチのターゲット値を格納するリスト

    val_loss = 0  # Evaluation loss

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass (inference)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

            # Evaluation
            val_loss += torch.sqrt(nn.MSELoss()(outputs, targets) + 1e-6).item()

    # すべてのバッチを結合し、必要に応じてテンソルに変換
    outputs = torch.cat(all_outputs).to(device)
    targets = torch.cat(all_targets).to(device)

    # サイズを一致させてからエラーを計算
    relative_errors = torch.abs((outputs - targets) / targets)

    total_samples += targets.size(0)  # バッチごとのサンプル数を加算

    if total_samples != 0:
        correct_predictions += torch.sum(relative_errors <= allowed_error).item()
        accuracy = correct_predictions / total_samples
        print(f"評価損失: {val_loss / len(dataloader):.6f}")
    else:
        print("No samples for inference.")

    # NVIDIA GPUの情報を取得し、出力に追加
    nvidia_smi_output = os.popen('nvidia-smi').read()
    print(nvidia_smi_output)


# GPUデバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのインスタンス化
model = NN()

# 推論データセットの準備
dataset = np.loadtxt("C:/Users/s1280/PycharmProjects/pythonProject1/EDFA_ML/DATASET/EDFA_1_15dB.csv", delimiter=",")
X_inference = torch.tensor(dataset[:, 1:86], dtype=torch.float32).to(device)
y_inference = torch.tensor(dataset[:, 86:], dtype=torch.float32).to(device)

# 推論データセットの作成
inference_dataset = MyDataset(X_inference, y_inference)
inference_loader = DataLoader(inference_dataset, batch_size=1024, shuffle=False)


def measure_throughput(model, dataloader):
    start_time = time.time()

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass (inference)
            outputs = model(inputs)

    end_time = time.time()
    elapsed_time = end_time - start_time

    throughput = len(dataloader.dataset) / elapsed_time
    print(f"スループット: {throughput:.2f} サンプル/秒")

    # キャッシュをクリア
    torch.cuda.empty_cache()

# FLOPsの計算
    macs, params = profile(model, inputs=(inputs,))
    print(f"FLOPs: {macs / 10 ** 9:.2f} GFLOPs")

# 推論の実行
measure_throughput(model, inference_loader)


inference_best_model(model, inference_loader, 'model_best_4-500.pth')

#inference_best_model(model, inference_loader, 'model_prunedCOL_4-485.pth')
#inference_best_model(model, inference_loader, 'model_pruned2_4-485.pth')


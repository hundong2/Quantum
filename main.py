# === 라이브러리 임포트 ===
import pennylane as qml
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch.nn import NLLLoss
import torchvision
from torch.utils.data import Subset, DataLoader
from matplotlib import pyplot as plt
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout2d, Linear
from torch import cat
from datetime import datetime
from tqdm import tqdm
import sys

# === 환경 설정 ===
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 데이터 준비 ===
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

train_ds = torchvision.datasets.FashionMNIST(
    "./", train=True, download=True,
    transform=transform)

test_ds = torchvision.datasets.FashionMNIST(
    "./", train=False, download=True,
    transform=transform)

train_mask = (train_ds.targets == 0) | (train_ds.targets == 6)
train_idx = torch.where(train_mask)[0]
train_ds.targets[train_ds.targets == 6] = 1

binary_train_ds = Subset(train_ds, train_idx)

train_loader = DataLoader(binary_train_ds, batch_size=16, shuffle=True)


# === 하이브리드 QML 모델 정의 ===
class BinaryClassifier(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)
        self.fc3 = Linear(1, 1)

        self.q_device = qml.device("default.qubit", wires=2)
        self.qnn_params = Parameter(torch.rand(8), requires_grad=True)
        self.obs = qml.PauliZ(0) @ qml.PauliZ(1)

        @qml.qnode(self.q_device)
        def circuit(x):
            qml.H(wires=0)
            qml.H(wires=1)
            qml.RZ(2. * x[0], wires=0)
            qml.RZ(2. * x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(2. * (torch.pi - x[0]) * (torch.pi - x[1]), wires=1)
            qml.CNOT(wires=[0, 1])

            qml.RY(2. * self.qnn_params[0], wires=0)
            qml.RY(2. * self.qnn_params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(2. * self.qnn_params[2], wires=0)
            qml.RY(2. * self.qnn_params[3], wires=1)
            qml.CNOT(wires=[1, 0])
            qml.RY(2. * self.qnn_params[4], wires=0)
            qml.RY(2. * self.qnn_params[5], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(2. * self.qnn_params[6], wires=0)
            qml.RY(2. * self.qnn_params[7], wires=1)

            return qml.expval(self.obs)

        self.qnn = circuit

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        q_out = torch.Tensor(x.size(0), 1).to(device)
        for i in range(x.size(0)):
            q_out[i] = self.qnn(x[i])

        x = self.fc3(q_out)

        return F.log_softmax(cat((x, 1 - x), -1), dim=1)

# === 모델 및 제약 조건 확인 ===
bc = BinaryClassifier()
bc.to(device)

total_params = sum(p.numel() for p in bc.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")

dummy_x = torch.tensor([0.0, 0.0], dtype=torch.float64)
specs = qml.specs(bc.qnn)(dummy_x)

assert specs["num_tape_wires"] <= 8, "❌ 큐빗 수 초과"
assert specs['resources'].depth <= 30, "❌ 회로 깊이 초과"
assert specs["num_trainable_params"] <= 60, "❌ 학습 퀀텀 파라미터 수 초과"
assert total_params <= 50000, "❌ 학습 전체 파라미터 수 초과"

print("✅ 회로 제약 통과 — 학습을 계속합니다")

# === 모델 학습 ===
optimizer = Adam(bc.parameters(), lr=0.001)
loss_func = NLLLoss()

epochs = 1
loss_history = []
bc.train()

for epoch in range(epochs):
    # tqdm의 출력이 버퍼에 남아있는 문제를 해결하기 위해 file 인자를 명시적으로 지정
    epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout)
    total_loss = []

    for data, target in epoch_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = bc(data)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = sum(total_loss) / len(total_loss)
    loss_history.append(avg_loss)
    print(f"\nTraining Epoch {epoch+1}: Average Loss: {avg_loss:.4f}")


# === 모델 추론 및 평가 ===
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
bc.eval()

all_preds, all_targets = [], []

with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Inference", file=sys.stdout):
        data, target = data.to(device), target.to(device)
        logits = bc(data)
        pred = logits.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

y_pred = np.array(all_preds)
y_true = np.array(all_targets)

# 평가 및 결과 저장
test_mask = (y_true == 0) | (y_true == 6)
y_pred_mapped = np.where(y_pred == 1, 6, y_pred)
acc = (y_pred_mapped[test_mask] == y_true[test_mask]).mean()
print(f"\nAccuracy (labels 0/6 only): {acc:.4f}")

now = datetime.now().strftime("%Y%m%d_%H%M%S")
y_pred_filename = f"y_pred_{now}.csv"
np.savetxt(y_pred_filename, y_pred_mapped, fmt="%d", header="prediction", comments="")

print(f"Prediction saved to {y_pred_filename}")

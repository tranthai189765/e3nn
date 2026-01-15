import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from e3nn import o3
import e3nn.nn as enn # Import module Neural Network của e3nn
import math
import random

# ==========================================
# 1. TẠO DỮ LIỆU (Tetris 3D: L vs I)
# ==========================================
class TetrisDataset(Dataset):
    def __init__(self, size=1000, rotated=False):
        self.data = []
        self.labels = []
        
        # Hình dáng cơ bản (4 điểm, mỗi điểm 3 tọa độ)
        shape_I = torch.tensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3]], dtype=torch.float32)
        shape_L = torch.tensor([[0,0,0], [0,0,1], [0,0,2], [1,0,0]], dtype=torch.float32)
        
        for _ in range(size):
            label = random.randint(0, 1)
            points = shape_I.clone() if label == 0 else shape_L.clone()
            
            # Thêm nhiễu và căn giữa
            points += torch.randn_like(points) * 0.05
            points = points - points.mean(dim=0)
            
            if rotated:
                rot = o3.rand_matrix() # Ma trận xoay ngẫu nhiên
                points = torch.matmul(points, rot.T)
            
            self.data.append(points)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ==========================================
# 2. MÔ HÌNH BASELINE (MLP Thường)
# ==========================================
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1) 
        return self.net(x)

# ==========================================
# 3. MÔ HÌNH EQUIVARIANT (E3NN) - ĐÃ SỬA LỖI
# ==========================================
class EquivariantModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ========= Irreps =========
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)   # edge feature
        self.irreps_hidden = o3.Irreps("8x0e + 4x1o")

        # ========= Message Passing =========
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=o3.Irreps("1x0e"),
            irreps_out=self.irreps_hidden,
        )

        self.act = enn.NormActivation(self.irreps_hidden, torch.relu)
        self.linear = o3.Linear(self.irreps_hidden, self.irreps_hidden)

        # ========= Invariant Readout =========
        self.norm = o3.Norm(self.irreps_hidden, squared=True)

        self.classifier = nn.Sequential(
            nn.Linear(self.irreps_hidden.num_irreps, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, pos):
        """
        pos: [B, N=4, 3]
        """
        B, N, _ = pos.shape
        device = pos.device

        # ========= Pairwise relative vectors =========
        # r_ij = x_j - x_i
        ri = pos[:, :, None, :]           # [B, N, 1, 3]
        rj = pos[:, None, :, :]           # [B, 1, N, 3]
        rel = rj - ri                     # [B, N, N, 3]

        # bỏ self-loop (i == j)
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        rel = rel[:, mask].view(B, N, N-1, 3)

        # ========= Spherical Harmonics on edges =========
        rel_flat = rel.reshape(-1, 3)
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel_flat,
            normalize=True,
            normalization="component"
        )

        # ========= Tensor Product (messages) =========
        ones = torch.ones(sh.shape[0], 1, device=device)
        msg = self.tp(sh, ones)
        msg = self.act(msg)
        msg = self.linear(msg)

        # ========= Aggregate messages =========
        msg = msg.view(B, N, N-1, -1)
        x = msg.sum(dim=2)    # sum over neighbors → [B, N, irreps_hidden]

        # ========= Invariant Readout =========
        x_inv = self.norm(x)  # [B, N, num_irreps]
        x_inv = x_inv.sum(dim=1)  # sum over points → [B, num_irreps]

        return self.classifier(x_inv)

# ==========================================
# 4. CHẠY THỰC NGHIỆM
# ==========================================
def train_and_evaluate(model_name, train_loader, test_loader, device):
    if model_name == "Baseline":
        model = BaselineMLP().to(device)
    else:
        model = EquivariantModel().to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Training {model_name} ---")
    model.train()
    for epoch in range(100): 
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Acc = {100 * correct / total:.1f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    acc = 100 * correct / total
    print(f">>> KẾT QUẢ TEST (Dữ liệu bị xoay): {acc:.2f}%")
    return acc

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
train_set = TetrisDataset(size=1000, rotated=False)
test_set = TetrisDataset(size=500, rotated=True)   
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# Run
acc_baseline = train_and_evaluate("Baseline", train_loader, test_loader, device)
acc_e3nn = train_and_evaluate("E3NN", train_loader, test_loader, device)

print("\n================ TỔNG KẾT ================")
print(f"Baseline (MLP) Accuracy trên tập xoay: {acc_baseline:.2f}%")
print(f"E3NN (Equivariant) Accuracy trên tập xoay: {acc_e3nn:.2f}%")
print("==========================================")
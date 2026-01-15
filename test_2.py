import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from e3nn import o3
import e3nn.nn as enn
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import os

# ==========================================
# 1. TẢI VÀ XỬ LÝ DATASET (MD17 - ASPIRIN)
# ==========================================
class GaussianRBF(nn.Module):
    def __init__(self, num_basis=8, r_max=5.0):
        super().__init__()
        centers = torch.linspace(0.0, r_max, num_basis)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, d):
        # d: [E]
        diff = d.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff ** 2)

class AspirinDataset(Dataset):
    def __init__(self, num_samples=None):
        url = "http://www.quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip"

        train_file = "aspirin_ccsd-train.npz"
        test_file = "aspirin_ccsd-test.npz"

        # Download nếu chưa có
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Đang tải MD17 Aspirin dataset từ {url} ...")
            r = requests.get(url)
            z = io.BytesIO(r.content)
            import zipfile
            with zipfile.ZipFile(z) as zip_ref:
                zip_ref.extractall(".")
            print("Tải xong!")

        # Load train + test
        train_data = np.load(train_file)
        test_data = np.load(test_file)

        # Concatenate
        R = np.concatenate([train_data["R"], test_data["R"]], axis=0)
        E = np.concatenate([train_data["E"], test_data["E"]], axis=0)
        z = train_data["z"]   # atom types giống nhau cho mọi mẫu

        self.R = torch.tensor(R, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)
        self.z = torch.tensor(z, dtype=torch.long)

        # Normalize energy
        self.E_mean = self.E.mean()
        self.E_std = self.E.std()
        self.E = (self.E - self.E_mean) / self.E_std

        # One-hot atom types: H, C, O
        type_map = {1: 0, 6: 1, 8: 2}
        z_idx = torch.tensor([type_map[a.item()] for a in self.z])
        self.node_attrs = torch.nn.functional.one_hot(z_idx, num_classes=3).float()

        # Limit samples if needed
        if num_samples is not None:
            self.R = self.R[:num_samples]
            self.E = self.E[:num_samples]

        print(f"Dataset Loaded: {self.R.shape[0]} samples, {self.R.shape[1]} atoms.")

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        return self.R[idx], self.node_attrs, self.E[idx]

# ==========================================
# 2. MODEL BASELINE (MLP trên tọa độ phẳng)
# ==========================================
class BaselineMLP(nn.Module):
    def __init__(self, n_atoms, atom_dim=3):
        super().__init__()
        input_dim = n_atoms * 3 + n_atoms * atom_dim  # pos + attrs
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, pos, node_attrs):
        batch_size = pos.shape[0]
        flat_pos = pos.view(batch_size, -1)
        flat_attrs = node_attrs.view(batch_size, -1)
        x = torch.cat([flat_pos, flat_attrs], dim=1)
        return self.net(x)

# ==========================================
# 3. MODEL E3NN (Equivariant)
# ==========================================
class E3NNModel(nn.Module):
    def __init__(self, max_radius=10.0):
        super().__init__()
        self.max_radius = max_radius
        
        # 1. Embedding loại nguyên tử (H, C, O -> 32 features)
        self.atom_emb = nn.Embedding(3, 32)
        
        # 2. Cấu hình Irreps
        self.irreps_in = o3.Irreps("32x0e")  # Input features
        self.irreps_hidden = o3.Irreps("64x0e + 32x1o") # Hidden features
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        self.irreps_out = o3.Irreps("1x0e") # Output energy

        # 3. Tensor Product
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_hidden,
            shared_weights=False,
            internal_weights=False
        )
        
        # 4. Radial Basis Function (learnable gamma)
        self.number_of_basis = 20
        self.rbf = GaussianRBF(self.number_of_basis, self.max_radius)
        self.fc_radial = nn.Sequential(
            nn.Linear(self.number_of_basis, 32),
            nn.ReLU(),
            nn.Linear(32, self.tp.weight_numel)
        )
        
        # 5. Self linear
        self.self_linear = o3.Linear(self.irreps_in, self.irreps_hidden)
        
        # 6. Output layers
        self.act = enn.NormActivation(self.irreps_hidden, torch.relu)
        self.readout = o3.Linear(self.irreps_hidden, self.irreps_out)

    def forward(self, pos, node_attrs):
        B, N, _ = pos.shape
        device = pos.device
        
        # --- 1. Node Features ---
        z_idx = node_attrs.argmax(dim=-1).long()
        h = self.atom_emb(z_idx)
        
        pos_flat = pos.view(B*N, 3)
        h_flat = h.view(B*N, -1)
        
        # --- 2. Tạo Đồ Thị (Fully Connected per Molecule) ---
        row_local = torch.arange(N, device=device).repeat_interleave(N)
        col_local = torch.arange(N, device=device).repeat(N)
        batch_offsets = torch.arange(B, device=device) * N
        
        row = (row_local.unsqueeze(0) + batch_offsets.unsqueeze(1)).view(-1)
        col = (col_local.unsqueeze(0) + batch_offsets.unsqueeze(1)).view(-1)
        
        mask = row != col
        row = row[mask]
        col = col[mask]
        
        # --- 3. Edge Features ---
        edge_vec = pos_flat[row] - pos_flat[col]
        edge_len = edge_vec.norm(dim=-1)
        
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, 
            edge_vec, 
            normalize=True, 
            normalization='component'
        )
        
        edge_radial = self.rbf(edge_len)
        
        # Map Radial ra đúng số weights mà TP cần
        tp_weights = self.fc_radial(edge_radial)
        
        # --- 4. Message Passing ---
        node_in = h_flat[col]
        messages = self.tp(node_in, edge_sh, tp_weights)
        
        # --- 5. Aggregation ---
        h_new = torch.zeros(B*N, messages.shape[1], device=device)
        h_new.index_add_(0, row, messages)
        
        # Add self contribution
        h_new = h_new + self.self_linear(h_flat)
        
        # --- 6. Readout ---
        h_new = self.act(h_new)
        atomic_energy = self.readout(h_new)
        
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)
        total_energy = torch.zeros(B, 1, device=device)
        total_energy.index_add_(0, batch_idx, atomic_energy)
        
        return total_energy

# ==========================================
# 4. EXPERIMENT LOOP (Data Efficiency)
# ==========================================
def run_experiment(device):
    # Load full data (lấy 1500 mẫu để tiết kiệm RAM, Aspirin gốc rất nặng)
    full_dataset = AspirinDataset(num_samples=1500)
    
    # Tập Test cố định (500 mẫu cuối)
    test_idx = list(range(1000, 1500))
    test_set = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=32)
    
    # Các mốc dữ liệu train cần thử nghiệm (loại bỏ 1500 để tránh overlap)
    train_sizes = [50, 100, 500, 1000]
    
    results = {
        "Baseline": [],
        "E3NN": []
    }
    
    criterion = nn.L1Loss() # Mean Absolute Error (MAE)
    
    print("\nBẮT ĐẦU THỰC NGHIỆM DATA EFFICIENCY...")
    print(f"Device: {device}")
    
    for size in train_sizes:
        print(f"\n>>> Training with N={size} samples")
        
        # Tạo tập train con
        train_idx = list(range(0, size))
        train_sub = Subset(full_dataset, train_idx)
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        
        # --- Train Baseline ---
        model_base = BaselineMLP(n_atoms=full_dataset.R.shape[1]).to(device)
        opt_base = optim.Adam(model_base.parameters(), lr=1e-3)
        
        # Train Loop (100 epochs)
        for epoch in range(100):
            model_base.train()
            for r, z, e in train_loader:
                r, z, e = r.to(device), z.to(device), e.to(device)
                opt_base.zero_grad()
                pred = model_base(r, z)
                loss = criterion(pred, e)
                loss.backward()
                opt_base.step()
        
        # Test Baseline
        model_base.eval()
        loss_sum = 0
        with torch.no_grad():
            for r, z, e in test_loader:
                r, z, e = r.to(device), z.to(device), e.to(device)
                pred = model_base(r, z)
                loss_sum += criterion(pred, e).item() * r.size(0)
        mae_base = loss_sum / len(test_set)
        results["Baseline"].append(mae_base)
        print(f"   Baseline Test MAE: {mae_base:.4f}")

        # --- Train E3NN ---
        model_e3nn = E3NNModel().to(device)
        opt_e3nn = optim.Adam(model_e3nn.parameters(), lr=1e-4)
        
        for epoch in range(300):
            model_e3nn.train()
            for r, z, e in train_loader:
                r, z, e = r.to(device), z.to(device), e.to(device)
                opt_e3nn.zero_grad()
                pred = model_e3nn(r, z)
                loss = criterion(pred, e)
                loss.backward()
                opt_e3nn.step()
                
        # Test E3NN
        model_e3nn.eval()
        loss_sum = 0
        with torch.no_grad():
            for r, z, e in test_loader:
                r, z, e = r.to(device), z.to(device), e.to(device)
                pred = model_e3nn(r, z)
                loss_sum += criterion(pred, e).item() * r.size(0)
        mae_e3nn = loss_sum / len(test_set)
        results["E3NN"].append(mae_e3nn)
        print(f"   E3NN Test MAE:     {mae_e3nn:.4f}")

    return train_sizes, results

# ==========================================
# 5. VẼ BIỂU ĐỒ (LOG-LOG PLOT)
# ==========================================
def plot_results(train_sizes, results):
    plt.figure(figsize=(8, 6))
    
    # Log-Log Plot
    plt.loglog(train_sizes, results["Baseline"], marker='o', label='Baseline (Invariant/MLP)', linestyle='--')
    plt.loglog(train_sizes, results["E3NN"], marker='s', label='E3NN (Equivariant)', linewidth=2)
    
    plt.xlabel('Number of Training Samples (Log Scale)')
    plt.ylabel('Test MAE Error (Log Scale)')
    plt.title('Data Efficiency: E3NN vs Baseline on MD17 (Aspirin)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.savefig('data_efficiency_loglog.png')
    print("Đã lưu biểu đồ: data_efficiency_loglog.png")
    plt.show()

# MAIN
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sizes, res = run_experiment(device)
    plot_results(sizes, res)
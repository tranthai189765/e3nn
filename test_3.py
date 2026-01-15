# qm9_e3nn_experiment.py
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----- optional imports with nice error messages -----
try:
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import radius_graph
    from torch_scatter import scatter_add
except Exception as e:
    raise RuntimeError(
        "Missing PyG / torch_scatter. Please install torch_geometric and torch_scatter.\n"
        "Example install (Linux, matching your torch/cuda):\n"
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html\n"
        "pip install torch-geometric -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html\n"
        "See https://pytorch-geometric.readthedocs.io/ for details."
    )

try:
    from e3nn import o3
    import e3nn.nn as enn
except Exception as e:
    raise RuntimeError("Missing e3nn. Install via `pip install e3nn` (matching your torch).")

# =======================================
# Config (thay đổi tùy máy / mục tiêu)
# =======================================
TARGET_IDX = 7          # QM9 target index for U0 (Internal energy at 0K). See PyG QM9 docs. :contentReference[oaicite:1]{index=1}
ROOT = "./qm9_data"
NUM_SAMPLES = 20000     # số mẫu lấy ra (10k-30k hợp lý). Set None để lấy toàn bộ (~130k)
BATCH_SIZE = 1024
MAX_RADIUS = 5.0        # radius graph cutoff (Å)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_BASELINE = 100    # epochs cho baseline (có thể tăng)
EPOCHS_E3NN = 100       # epochs cho e3nn (mỗi epoch chậm hơn, tùy chỉnh)
LR_BASELINE = 1e-6
LR_E3NN = 3e-4
SEED = 42

torch.manual_seed(SEED)

# =======================================
# 1. Load QM9 dataset (PyG)
# =======================================
print("Loading QM9 dataset (this may download ~300MB)...")
dataset = QM9(root=ROOT)

if NUM_SAMPLES is not None and NUM_SAMPLES < len(dataset):
    dataset = dataset[:NUM_SAMPLES]

# Split: train / val / test (simple)
N = len(dataset)
n_train = int(0.85 * N)
n_val = int(0.05 * N)
n_test = N - n_train - n_val

train_dataset = dataset[:n_train]
val_dataset = dataset[n_train:n_train+n_val]
test_dataset = dataset[n_train+n_val:]

train_loader = GeoDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = GeoDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = GeoDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_y = []
for data in train_dataset:
    y = data.y[:, TARGET_IDX]
    if not torch.isnan(y):
        all_y.append(y.item())

all_y = torch.tensor(all_y)
Y_MEAN = all_y.mean()
Y_STD = all_y.std()

print(f"[INFO] Target mean={Y_MEAN:.4f}, std={Y_STD:.4f}")

print(f"Dataset sizes => total: {N}, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
print(f"Device: {DEVICE}")

# =======================================
# 2. Baseline (Invariant) simple GNN
# =======================================
class BaselineInvariantGNN(nn.Module):
    def __init__(self, atom_embed=64, hidden=128, max_radius=MAX_RADIUS):
        super().__init__()
        # atom embedding from atomic number (data.z in QM9)
        self.atom_emb = nn.Embedding(100, atom_embed)
        # MLP to produce messages from (node_j_emb, dist_rbf)
        self.rbf_means = torch.linspace(0.0, max_radius, steps=16)
        self.gamma = nn.Parameter(torch.tensor(10.0))
        self.msg_mlp = nn.Sequential(
            nn.Linear(atom_embed + len(self.rbf_means), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(atom_embed + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def rbf(self, d):
        # d: [E]
        centers = self.rbf_means.to(d.device)
        diff = d.unsqueeze(-1) - centers
        return torch.exp(-self.gamma * diff ** 2)

    def forward(self, data):
        # data.pos: [num_nodes,3]
        # data.z: [num_nodes] with atomic numbers (or given)
        pos = data.pos
        z = (data.z.long()).squeeze() if data.z is not None else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        batch = data.batch
        x = self.atom_emb(z)  # [num_nodes, atom_embed]

        edge_index = radius_graph(pos, r=MAX_RADIUS, batch=batch, max_num_neighbors=64)
        row, col = edge_index  # edge i <- j (row = target, col = source)

        edge_vec = pos[row] - pos[col]
        edge_len = edge_vec.norm(dim=-1)
        edge_rbf = self.rbf(edge_len)  # [E, R]

        node_j = x[col]
        msg_in = torch.cat([node_j, edge_rbf], dim=-1)
        messages = self.msg_mlp(msg_in)

        # aggregate messages to target nodes
        agg = scatter_add(messages, row, dim=0, dim_size=x.size(0))

        node_feat = torch.cat([x, agg], dim=-1)
        node_out = self.node_mlp(node_feat)

        # global pooling (sum)
        mol_feat = scatter_add(node_out, batch, dim=0)
        out = self.readout(mol_feat).squeeze(-1)  # [batch_size]
        return out

class InteractionBlock(nn.Module):
    def __init__(self, irreps_node, irreps_sh, irreps_out, max_radius):
        super().__init__()

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_node,
            irreps_sh,
            irreps_out,
            shared_weights=False,
            internal_weights=False
        )

        self.num_basis = 20
        self.register_buffer(
            "centers",
            torch.linspace(0.0, max_radius, self.num_basis)
        )
        self.gamma = nn.Parameter(torch.tensor(8.0))

        self.fc_radial = nn.Sequential(
            nn.Linear(self.num_basis, 64),
            nn.ReLU(),
            nn.Linear(64, self.tp.weight_numel)
        )

        self.self_linear = o3.Linear(irreps_node, irreps_out)
        self.act = enn.NormActivation(irreps_out, torch.relu)

    def rbf(self, d):
        diff = d.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff ** 2)

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_len = edge_vec.norm(dim=-1)

        edge_sh = o3.spherical_harmonics(
            self.tp.irreps_in2, edge_vec,
            normalize=True, normalization="component"
        )

        edge_radial = self.rbf(edge_len)
        tp_weights = self.fc_radial(edge_radial)

        msg = self.tp(h[col], edge_sh, tp_weights)
        agg = scatter_add(msg, row, dim=0, dim_size=h.shape[0])

        h_new = agg + self.self_linear(h)
        return self.act(h_new)

# =======================================
# 3. E3NN Equivariant Model (simple message passing)
# =======================================
class E3NNQM9(nn.Module):
    def __init__(self, max_radius=MAX_RADIUS):
        super().__init__()
        self.max_radius = max_radius

        self.irreps_in = o3.Irreps("64x0e")
        self.irreps_hidden = o3.Irreps("16x0e + 8x1o")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        self.irreps_out = o3.Irreps("1x0e")

        self.atom_emb = nn.Embedding(100, self.irreps_in.dim)

        self.block1 = InteractionBlock(
            self.irreps_in, self.irreps_sh, self.irreps_hidden, max_radius
        )
        self.block2 = InteractionBlock(
            self.irreps_hidden, self.irreps_sh, self.irreps_hidden, max_radius
        )
        self.block3 = InteractionBlock(
            self.irreps_hidden, self.irreps_sh, self.irreps_hidden, max_radius
        )

        self.readout = o3.Linear(self.irreps_hidden, self.irreps_out)

    def forward(self, data):
        pos = data.pos
        z = data.z.long()
        batch = data.batch

        h = self.atom_emb(z)

        edge_index = radius_graph(
            pos, r=self.max_radius, batch=batch, max_num_neighbors=64
        )

        h = self.block1(h, pos, edge_index)
        h = self.block2(h, pos, edge_index)
        h = self.block3(h, pos, edge_index)

        atomic_energy = self.readout(h).squeeze(-1)
        mol_energy = scatter_add(atomic_energy, batch, dim=0)
        return mol_energy

# =======================================
# 4. Training / Evaluation utilities
# =======================================
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.L1Loss(reduction='none')
    total_loss = 0.0
    n_total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            y = (data.y[:, TARGET_IDX].to(device).squeeze() - Y_MEAN) / Y_STD  # select U0
            pred = model(data)
            # y might be NaN for some entries in QM9; filter
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            l = loss_fn(pred[mask], y[mask]).sum().item()
            total_loss += l
            n_total += mask.sum().item()
    return total_loss / max(1, n_total)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.L1Loss()
    total = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        y = (data.y[:, TARGET_IDX].to(device).squeeze() - Y_MEAN) / Y_STD
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred[mask], y[mask])
        loss.backward()
        optimizer.step()
        total += loss.item() * mask.sum().item()
        n += mask.sum().item()
    return total / max(1, n)

# =======================================
# 5. Run experiment (train baseline then e3nn)
# =======================================
def run_experiment():
    # Baseline
    print("\n--- Training Baseline Invariant GNN ---")
    baseline = BaselineInvariantGNN().to(DEVICE)
    opt_b = optim.Adam(baseline.parameters(), lr=LR_BASELINE)
    history = {"baseline_train": [], "baseline_val": []}

    for epoch in range(1, EPOCHS_BASELINE + 1):
        t0 = time.time()
        train_mae = train_one_epoch(baseline, train_loader, opt_b, DEVICE)
        val_mae = evaluate(baseline, val_loader, DEVICE)
        history["baseline_train"].append(train_mae)
        history["baseline_val"].append(val_mae)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Baseline] Epoch {epoch:03d}/{EPOCHS_BASELINE}  train_MAE={train_mae:.4f}  val_MAE={val_mae:.4f}  time={time.time()-t0:.1f}s")

    test_mae_baseline = evaluate(baseline, test_loader, DEVICE)
    print(f"Baseline Test MAE (target idx {TARGET_IDX}): {test_mae_baseline:.4f}")

    # E3NN model
    print("\n--- Training E3NN Equivariant Model ---")
    e3nn_model = E3NNQM9(max_radius=MAX_RADIUS).to(DEVICE)
    opt_e = optim.Adam(e3nn_model.parameters(), lr=LR_E3NN)
    history["e3nn_train"] = []
    history["e3nn_val"] = []

    for epoch in range(1, EPOCHS_E3NN + 1):
        t0 = time.time()
        train_mae = train_one_epoch(e3nn_model, train_loader, opt_e, DEVICE)
        val_mae = evaluate(e3nn_model, val_loader, DEVICE)
        history["e3nn_train"].append(train_mae)
        history["e3nn_val"].append(val_mae)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[E3NN] Epoch {epoch:03d}/{EPOCHS_E3NN}  train_MAE={train_mae:.4f}  val_MAE={val_mae:.4f}  time={time.time()-t0:.1f}s")

    test_mae_e3nn = evaluate(e3nn_model, test_loader, DEVICE)
    print(f"E3NN Test MAE (target idx {TARGET_IDX}): {test_mae_e3nn:.4f}")

    # Save results & plot
    return history, test_mae_baseline, test_mae_e3nn

# =======================================
# 6. Plot helper
# =======================================
def plot_history(history):
    plt.figure(figsize=(8,6))
    # Baseline
    plt.plot(history['baseline_val'], label='Baseline Val MAE')
    # E3NN (if present)
    if 'e3nn_val' in history:
        plt.plot(history['e3nn_val'], label='E3NN Val MAE')
    plt.xlabel('Epochs (val per epoch)')
    plt.ylabel('MAE')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title('Validation MAE (QM9, target=U0)')
    plt.savefig('qm9_e3nn_history.png')
    print("Saved plot: qm9_e3nn_history.png")
    plt.show()

# =======================================
# Main
# =======================================
if __name__ == "__main__":
    hist, baseline_test, e3nn_test = run_experiment()
    print("\nFINAL RESULTS:")
    print(f" Baseline Test MAE: {baseline_test:.4f}")
    print(f" E3NN   Test MAE: {e3nn_test:.4f}")
    plot_history(hist)

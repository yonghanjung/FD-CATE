import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0); np.random.seed(0)

# ----------------------------
# 1) Toy FD data generator
# ----------------------------
def make_fd_toy(n=20000, d=5):
    C = np.random.normal(size=(n, d))
    U = np.random.normal(size=(n, 1))  # unobserved confounder

    # Treatment assignment X|C,U
    logit_x = 0.8*(C[:,0:1]) + 0.6*(C[:,1:2]) + 1.0*U
    p_x = 1/(1+np.exp(-logit_x))
    X = (np.random.rand(n,1) < p_x).astype(np.float32)

    # Mediator Z|X,C  (front-door path X->Z, *no* direct U->Z to satisfy FD)
    logit_z = 1.2*X + 0.5*(C[:,2:3]) - 0.5*(C[:,3:4])
    p_z = 1/(1+np.exp(-logit_z))
    Z = (np.random.rand(n,1) < p_z).astype(np.float32)

    # Outcome Y|Z,X,C,U  (allow direct U->Y, and X->Y for heterog.)
    mu = 0.5*Z + 0.2*X + 0.5*np.tanh(C[:,0:1]-C[:,1:2]) + 0.9*U
    Y = mu + 0.5*np.random.normal(size=(n,1))

    C = C.astype(np.float32); Y = Y.astype(np.float32)
    return C, X, Z, Y

# ----------------------------
# 2) Multi-task model (shared f(C) and task-specific heads)
#    Heads implement:
#      e(x|C) via softmax over x∈{0,1}
#      q(z|x,C) via softmax over z∈{0,1} given x
#      m(z,x,C) via two-branch regression over (z,x)
# ----------------------------
class Backbone(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU()
        )
    def forward(self, C):
        return self.net(C)

class HeadE(nn.Module):
    # logits over x∈{0,1}
    def __init__(self, h):
        super().__init__()
        self.fc = nn.Linear(h, 2)
    def forward(self, feat):
        return self.fc(feat)  # (N,2)

class HeadQ(nn.Module):
    # logits over z∈{0,1} for each x separately (two heads conditional on x)
    def __init__(self, h):
        super().__init__()
        self.fc0 = nn.Linear(h, 2)  # q(.|x=0,C)
        self.fc1 = nn.Linear(h, 2)  # q(.|x=1,C)
    def forward(self, feat):
        return self.fc0(feat), self.fc1(feat)

class HeadM(nn.Module):
    # regress Y for each (z,x) pair with simple heads (four scalar regressors)
    def __init__(self, h):
        super().__init__()
        self.f00 = nn.Linear(h, 1)
        self.f01 = nn.Linear(h, 1)
        self.f10 = nn.Linear(h, 1)
        self.f11 = nn.Linear(h, 1)
    def forward(self, feat):
        return self.f00(feat), self.f01(feat), self.f10(feat), self.f11(feat)

class LobsterPlugIn(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.backbone = Backbone(d, h)
        self.head_e = HeadE(h)
        self.head_q = HeadQ(h)
        self.head_m = HeadM(h)

    def forward_e_logits(self, C):
        feat = self.backbone(C); return self.head_e(feat)

    def forward_q_logits(self, C):
        feat = self.backbone(C); return self.head_q(feat)

    def forward_m_values(self, C):
        feat = self.backbone(C); return self.head_m(feat)

# ----------------------------
# 3) Losses for multitask training
# ----------------------------
def cross_entropy_logits(logits, target01):
    # target01 ∈ {0,1} (float or long), logits shape (N,2)
    return F.cross_entropy(logits, target01.squeeze().long())

def mse(pred, target):
    return F.mse_loss(pred, target)

# ----------------------------
# 4) Training loop
# ----------------------------
def train_model(C, X, Z, Y, epochs=10, batch=128, lr=1e-3, h=64, verbose=True, seed = 1234):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N, d = C.shape
    model = LobsterPlugIn(d, h).to(device)

    ds = TensorDataset(torch.from_numpy(C),
                       torch.from_numpy(X),
                       torch.from_numpy(Z),
                       torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        tot = 0.0; nb = 0
        for Cb, Xb, Zb, Yb in dl:
            Cb = Cb.to(device); Xb = Xb.to(device); Zb = Zb.to(device); Yb = Yb.to(device)

            # e(x|C): CE loss on X
            e_logits = model.forward_e_logits(Cb)
            loss_e = cross_entropy_logits(e_logits, Xb)

            # q(z|x,C): CE loss on Z with the appropriate head by x
            q0_logits, q1_logits = model.forward_q_logits(Cb)
            # pick logits based on X:
            qb = torch.stack([q0_logits, q1_logits], dim=0)          # (2,N,2)
            idx = Xb.squeeze().long()                                 # (N,)
            gather_q = qb[ idx, torch.arange(len(idx)), : ]           # (N,2)
            loss_q = cross_entropy_logits(gather_q, Zb)

            # m(z,x,C): MSE on Y with the head for the observed (z,x)
            m00, m01, m10, m11 = model.forward_m_values(Cb)
            # select by (Z,X)
            pred_m = (
                (1-Zb)*(1-Xb)*m00 + (1-Zb)*Xb*m01 + Zb*(1-Xb)*m10 + Zb*Xb*m11
            )
            loss_m = mse(pred_m, Yb)

            loss = loss_e + loss_q + loss_m
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose and (ep % 2 == 0 or ep == 1):
            print(f"[epoch {ep:02d}] loss={tot/nb:.4f}")
    return model

# ----------------------------
# 5) Plug-in τ̂(C): uses learned \hat e, \hat q, \hat m
# ----------------------------
@torch.no_grad()
def tau_hat_C(model, C):
    device = next(model.parameters()).device
    Ct = torch.from_numpy(C).to(device)

    # e(x|C)
    e_logits = model.forward_e_logits(Ct)            # (N,2)
    e_prob = F.softmax(e_logits, dim=1)              # columns: x=0, x=1
    e0, e1 = e_prob[:,0:1], e_prob[:,1:2]

    # q(z|x,C)
    q0_logits, q1_logits = model.forward_q_logits(Ct)
    q0 = F.softmax(q0_logits, dim=1)                 # (N,2), z=0,1
    q1 = F.softmax(q1_logits, dim=1)
    q0_z0, q0_z1 = q0[:,0:1], q0[:,1:2]
    q1_z0, q1_z1 = q1[:,0:1], q1[:,1:2]

    # m(z,x,C)
    m00, m01, m10, m11 = model.forward_m_values(Ct)  # (N,1) each

    # sum_{z,x} {q(z|1,C)-q(z|0,C)} e(x|C) m(z,x,C)
    term = 0
    term += (q1_z0 - q0_z0) * e0 * m00
    term += (q1_z1 - q0_z1) * e0 * m10
    term += (q1_z0 - q0_z0) * e1 * m01
    term += (q1_z1 - q0_z1) * e1 * m11
    return term.cpu().numpy()  # (N,1)

# ----------------------------
# 6) Run: generate data, fit, compute τ̂(C)
# ----------------------------
if __name__ == "__main__":
    C, X, Z, Y = make_fd_toy(n=20000, d=6)
    model = train_model(C, X, Z, Y, epochs=15, batch=1024, lr=2e-3, h=64, verbose=False)

    tau_hat = tau_hat_C(model, C[:10000])
    print("\nτ̂(C) for 10 samples:\n", np.round(tau_hat.squeeze(), 3))

#!/usr/bin/env python
# coding: utf-8

# In[12]:


from random import shuffle
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
pca = PCA(n_components=2)

mnist = fetch_openml('mnist_784')
X_orig, y_orig = mnist["data"], mnist["target"]

X_orig = mms.fit_transform(np.array(X_orig).astype(np.float32))
y_orig = np.array(y_orig).astype(np.uint8)
target_digit1 = 3
target_digit2 = 8

target_digit1_xdata = X_orig[y_orig == target_digit1]
target_digit2_xdata = X_orig[y_orig == target_digit2]
target_digit1_ydata = y_orig[y_orig == target_digit1]
target_digit2_ydata = y_orig[y_orig == target_digit2]
X = np.concatenate((target_digit1_xdata, target_digit2_xdata), axis=0)
y = np.concatenate((target_digit1_ydata, target_digit2_ydata), axis=0)
y = np.where(y == target_digit1, 0, 1)
X = pca.fit_transform(X)
X = mms.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

pca.fit(X_train)

print(X_train.shape, X_test.shape, X_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

# In[16]:


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# assuming X_train, y_train, X_val, y_val, X_test, y_test are numpy arrays
def to_tensors(X, y):
    X_t = torch.from_numpy(np.asarray(X)).float()         # [N, D]
    y_t = torch.from_numpy(np.asarray(y)).long().view(-1) # [N]
    return X_t, y_t

Xtr_t, ytr_t = to_tensors(X_train, y_train)
Xva_t, yva_t = to_tensors(X_val,   y_val)
Xte_t, yte_t = to_tensors(X_test,  y_test)

train_ds = TensorDataset(Xtr_t, ytr_t)
val_ds   = TensorDataset(Xva_t, yva_t)
test_ds  = TensorDataset(Xte_t, yte_t)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=512)
test_loader  = DataLoader(test_ds,  batch_size=512)

# model wiring with the SVM head from before:
D = Xtr_t.shape[1]
K = int(max(ytr_t.max(), yva_t.max(), yte_t.max()).item() + 1)


# In[17]:


# linear_l2_svm.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
train_df = Dataset()
train_load = DataLoader(X_train, batch_size=64, shuffle=True)


# In[18]:


# ----------- Model -----------

class SVMClassifier(nn.Module):
    """
    Linear L2-SVM head (squared hinge) with optional feature extractor.
    - Multiclass via one-vs-rest (K linear scores).
    - Bias is NOT regularized (as in SVMs).
    """
    def __init__(self, in_features: int, num_classes: int, feature_extractor: nn.Module | None = None, bias: bool = True):
        super().__init__()
        self.backbone = feature_extractor if feature_extractor is not None else nn.Identity()
        self.head = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        scores = self.head(z)  # shape: [N, K]
        return scores


# ----------- Loss (primal L2-SVM) -----------

def _make_ovr_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    y: [N] with class indices in [0, K-1]
    returns Y_bin: [N, K] with +1 for true class, -1 otherwise (one-vs-rest targets)
    """
    N = y.shape[0]
    Y_bin = -torch.ones((N, num_classes), device=y.device, dtype=torch.float32)
    Y_bin.scatter_(1, y.view(-1, 1), 1.0)
    return Y_bin


def l2_svm_primal_loss(scores: torch.Tensor, y: torch.Tensor, W: torch.Tensor, C: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    """
    Loss = 0.5 * ||W||_F^2 + C * mean( clamp(1 - y_bin * scores, 0)^2 )
    - scores: [N, K] raw class scores (no softmax)
    - y: [N] ground-truth class indices in [0..K-1]
    - W: weight matrix of the linear head, shape [K, D] (bias not regularized)
    """
    assert scores.dim() == 2 and y.dim() == 1
    N, K = scores.shape
    Y_bin = _make_ovr_targets(y, K)                              # [N, K] in {+1, -1}
    margins = 1.0 - Y_bin * scores                               # [N, K]
    hinge = torch.clamp(margins, min=0.0)
    data_loss = (hinge ** 2)
    if reduction == "mean":
        data_loss = data_loss.mean()                              # average over N*K
    elif reduction == "sum":
        data_loss = data_loss.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    reg = 0.5 * (W ** 2).sum()                                   # L2 on weights ONLY
    return reg + C * data_loss


# ----------- Metrics -----------

@torch.no_grad()
def top1_accuracy(scores: torch.Tensor, y: torch.Tensor) -> float:
    pred = scores.argmax(dim=1)
    return (pred == y).float().mean().item()


# ----------- Training / Eval Loops -----------

def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                C: float = 1.0) -> tuple[float, float]:
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device).long()

        scores = model(x)
        loss = l2_svm_primal_loss(scores, y, W=model.head.weight, C=C, reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc  += top1_accuracy(scores, y)
        n_batches    += 1

    return running_loss / n_batches, running_acc / n_batches


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             C: float = 1.0) -> tuple[float, float]:
    model.eval()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        scores = model(x)
        loss = l2_svm_primal_loss(scores, y, W=model.head.weight, C=C, reduction="mean")

        running_loss += loss.item()
        running_acc  += top1_accuracy(scores, y)
        n_batches    += 1

    return running_loss / n_batches, running_acc / n_batches


# ----------- Example wiring (fill in your own loaders/backbone) -----------

if __name__ == "__main__":
    """
    Replace `train_loader`/`val_loader` with your DataLoaders.
    If you have a backbone that outputs D-dim features, pass it to SVMClassifier.
    For raw tabular features, use backbone=None and set in_features accordingly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = [to_tensors(X_train, y_train)]
    val_loader = [to_tensors(X_val, y_val)]

    # --- Example placeholders (replace) ---
    D = 2          # feature dimensionality
    K = 2           # number of classes

    backbone = nn.Identity()  # or your feature extractor module
    model = SVMClassifier(in_features=D, num_classes=K, feature_extractor=backbone, bias=True).to(device)

    # IMPORTANT: set weight_decay=0. We add the exact SVM regularizer explicitly.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=1e-2, weight_decay=0.0)

    C = 1.0  # trade-off between margin term and hinge penalty; tune per dataset

    for epoch in range(1, 201):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, C=C)
        va_loss, va_acc = evaluate(model, val_loader, device, C=C)
        print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")




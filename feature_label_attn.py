import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import os
from datetime import datetime
from sklearn.datasets import make_multilabel_classification

from mlp_att import tabpfn_pred_probas, get_dataset, evaluate_multilabel

class JointFeatureLabelAttn(nn.Module):
    """
    同时建模：
    - 标签自注意力（label-label 关系）
    - 标签对特征的交叉注意力（label-feature 关系）
    输入：
      - features: 原始样本特征 (B, n_features)
      - probas: 一级模型的每标签概率 (B, n_labels)
    输出：
      - logits: (B, n_labels)
    """
    def __init__(
        self,
        n_features: int,
        n_labels: int, 
        d_model: int = 64,
        n_heads: int = 4,
        n_feature_tokens: int = 6,
        h_feature: int = 128,
        h_label: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.n_feature_tokens = n_feature_tokens,
        self.d_model = d_model
        
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, h_feature),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_feature, n_feature_tokens * d_model),
        )
        
        self.label_proj = nn.Sequential(
            nn.Linear(1, h_label),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_label, d_model),
        )
        self.label_emb = nn.Parameter(torch.randn(n_labels, d_model))
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_ln = nn.LayerNorm(d_model)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.self_ln = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_ln = nn.LayerNorm(d_model)
        
        self.classifier = nn.Linear(d_model, 1)
        
    def forward(self, features: torch.Tensor, probas:torch.Tensor) -> torch.Tensor:
        """
        features:  (B, n_features), float32
        probas: (B, n_labels),   float32 in [0,1]
        return: logits (B, n_labels)
        """
        B = features.shape[0]
        feature_tokens = self.feature_proj(features)
        feature_tokens = feature_tokens.view(B, self.n_feature_tokens, self.d_model)
        
        z_label = self.label_proj(probas.unsqueeze(-1))
        z_label = z_label + self.label_emb.unsqueeze(0)
        
        cross_out, _ = self.cross_attn(query=z_label, key=feature_tokens, value=feature_tokens)
        z = self.cross_ln(z_label + cross_out)
        
        self_out, _ = self.self_attn(z, z, z)
        z = self.self_ln(z + self_out)
        
        z = z + self.ffn(z)
        z = self.ffn_ln(z)
        
        logits = self.classifier(z).squeeze(-1)
        return logits
    
class FeatureLabelDataset(Dataset):
    """
    同时提供样本特征与每标签先验概率（来自一级模型），以及目标 y。
    """
    def __init__(self, features: np.ndarray, probas: np.ndarray, y: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.probas = torch.tensor(probas, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.probas[idx], self.y[idx]
    
def gen_oof_with_features(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    生成与 OOF 概率对齐的特征与标签：(X_oof, y_oof, p_oof)
    便于用 OOF 概率训练二级模型，且与特征/标签按相同顺序对齐。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_oof, y_oof, p_oof = [], [], []
    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        p_te = tabpfn_pred_probas(X_tr, y_tr, X_te)
        X_oof.append(X_te)
        y_oof.append(y_te)
        p_oof.append(p_te)
    return np.vstack(X_oof), np.vstack(y_oof), np.vstack(p_oof)

def train_joint(
    model: nn.Module,
    features: np.ndarray,
    probas: np.ndarray,
    y: np.ndarray,
    optimizer,
    criterion,
    save_model: bool = False,
    epochs: int = 10,
    batch_size: int = 128,
    early_stopping_patience: int = 5,
    early_stopping_delta: float = 1e-4,
    validation_split: float = 0.15
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    feature_train, feature_val, probas_train, probas_val, y_train, y_val = train_test_split(
        features, probas, y, test_size=validation_split, random_state=42
    )
    print(f"train_joint: train_size={len(feature_train)}, val_size={len(feature_val)}, n_labels={probas.shape[1]}")

    train_ds = FeatureLabelDataset(feature_train, probas_train, y_train)
    val_ds = FeatureLabelDataset(feature_val, probas_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    best_val = float('inf')
    patience = 0
    best_state = None
    best_epoch = 0
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_batches = 0.0, 0
        for feats, probs, yb in train_loader:
            feats, probs, yb = feats.to(device), probs.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(feats, probs)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = train_loss / max(1, train_batches)
        
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for feats, probs, yb in val_loader:
                feats, probs, yb = feats.to(device), probs.to(device), yb.to(device)
                logits = model(feats, probs)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val - early_stopping_delta:
            best_val = avg_val_loss
            patience = 0
            best_epoch = epoch + 1
            best_state = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }
            print(f"    New best validation loss: {avg_val_loss:.4f}")
        else:
            patience += 1
            print(f"    No improvement for {patience} epochs")
            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if save_model and best_state is not None:
        ts = datetime.now().strftime("%m%d_%H%M")
        path = os.path.join('models', f'joint_model_best_epoch{best_epoch}_{ts}.pt')
        torch.save(best_state, path)
        print(f"Saved best joint model to {path}")
        
def predict_joint(model, X_train, y_train, X_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probas_n = tabpfn_pred_probas(X_train, y_train, X_test)
    probas_t = torch.tensor(probas_n, dtype=torch.float32, device=device)
    features_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    print(f"predict_joint: features_t.shape={features_t.shape}, probas_t.shape={probas_t.shape}")
    model.eval()
    with torch.no_grad():
        logits = model(features_t, probas_t)
        probs = torch.sigmoid(logits)

    print(f"predict_joint: produced probs shape={probs.shape}")
    return probs.cpu().numpy()

def main():
    X, Y = make_multilabel_classification(n_samples=200, n_features=20, n_classes=6,
                                          n_labels=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    # X_train, X_test, y_train, y_test = get_dataset(n_total_samples=200, test_split=0.15)
    n_features, n_labels = X_train.shape[1], y_train.shape[1]
    
    print(f"main: n_features={n_features}, n_labels={n_labels}")
    model = JointFeatureLabelAttn(
        n_features=n_features,
        n_labels=n_labels,
    )
    print(f"main: model params count = {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train, y_train, probas_train = gen_oof_with_features(X_train, y_train)
    print(f"main: after gen_oof_with_features X_train.shape={X_train.shape}, probas_train.shape={probas_train.shape}, y_train.shape={y_train.shape}")
    print("training start...")
    train_joint(
        model=model,
        features=X_train,
        probas=probas_train,
        y=y_train,
        optimizer=optimizer,
        criterion=criterion,
        save_model=False,
        epochs=30,
        batch_size=128
    )
    print("training over")
    
    y_test_probas = predict_joint(model, X_train, y_train, X_test)
    evaluate_multilabel(y_test, y_test_probas, obj_info='TabPFN+JointAttention')
    
if __name__ == '__main__':
    main()
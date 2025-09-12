import openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from tabpfn import TabPFNClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
from datetime import datetime
from utils import apply_thresholds, formalize_output_probas

class MLPWithAttention(nn.Module):
    def __init__(self, n_labels, d_model=32, hidden=8, n_heads=4):
        super().__init__()
        # 共享 MLP: 把每个标签的一级概率投影到 d_model 维
        self.embed_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_model),
        )
        # 可学习的标签嵌入
        self.label_emb = nn.Parameter(torch.randn(n_labels, d_model))
        # 自注意力层
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=0.1, batch_first=True)
        # 前馈层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        # 分类头
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, probs):
        # probs: (batch, n_labels)
        x = probs.unsqueeze(-1)  # (batch, n_labels, 1)
        z = self.embed_mlp(x)    # (batch, n_labels, d_model)
        z = z + self.label_emb.unsqueeze(0)  # 加上标签嵌入

        # Self-Attention
        attn_out, _ = self.attn(z, z, z)  # (batch, n_labels, d_model)
        z = z + attn_out
        z = z + self.ffn(z)

        # 分类头
        logits = self.classifier(z).squeeze(-1)  # (batch, n_labels)
        return logits

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def tabpfn_pred_probas(X_train, y_train, X_test):
    n_labels = y_train.shape[1]
    results = []
    classes_list = []
    for i in range(n_labels):
        target = y_train[:, i]
        model = TabPFNClassifier(random_state=42)
        model.fit(X_train, target)
        results.append(model.predict_proba(X_test))
        classes_list.append(model.classes_)
    return formalize_output_probas(results, n_labels, classes_list)

def gen_oof_data(X, y):
    kf = KFold()
    tabpfn_pred = []
    ground_truth = []
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        tabpfn_pred.append(tabpfn_pred_probas(X_train, y_train, X_test))
        ground_truth.append(y[test_index])
    return np.vstack(tabpfn_pred), np.vstack(ground_truth)

def train(model, X_train, y_train, optimizer, criterion, save_model=False, epochs=10, batch_size=128, 
          early_stopping_patience=5, early_stopping_delta=1e-4, validation_split=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42
    )
    
    train_ds = TabDataset(X_tr, y_tr)
    val_ds = TabDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stopped = False
    best_epoch = 0
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            # 保存最佳模型状态
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }
            print(f"    New best validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    No improvement for {patience_counter} epochs")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                early_stopped = True
                break
    print('training over')
    
    # 保存模型
    if save_model:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        if early_stopped and best_model_state is not None:
            # 保存早停时的最佳模型
            save_path = os.path.join('models', f'model_early_stop_epoch{best_epoch}_{timestamp}.pt')
            torch.save(best_model_state, save_path)
            print(f"Early stopped model saved to {save_path} (best epoch: {best_epoch}, val_loss: {best_val_loss:.4f})")
            
            # 恢复最佳模型状态
            model.load_state_dict(best_model_state['model_state_dict'])
            optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
        else:
            # 保存最终模型
            final_epoch = epoch + 1 if not early_stopped else epochs
            save_path = os.path.join('models', f'model_epoch{final_epoch}_{timestamp}.pt')
            torch.save({
                'epoch': final_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")
    
def load_model(model, optimizer=None, path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        files = glob.glob(os.path.join("models", "*.pt"))
        if not files:
            raise FileNotFoundError("No checkpoint files found in 'models' directory.")
        path = max(files, key=os.path.getmtime)  # latest by modification time

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # move optimizer state tensors to device
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    epoch = checkpoint.get("epoch", None)
    print(f"Loaded checkpoint '{path}' (epoch={epoch})")

def predict(model, X_train, y_train, X_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probs_pfn_n = tabpfn_pred_probas(X_train, y_train, X_test)
    probs_pfn_t = torch.tensor(probs_pfn_n, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        logits = model(probs_pfn_t)
        probs = torch.sigmoid(logits)
    
    return probs.cpu().numpy()

def evaluate_multilabel(y_true, y_prob, print_results=True, obj_info=None, thresholds=None):
    """
    计算多标签分类的评估指标
    
    参数:
        y_true: 真实标签，形状 (n_samples, n_labels)
        y_pred: 预测标签（0/1），形状 (n_samples, n_labels)  
        y_prob: 预测概率，形状 (n_samples, n_labels)
        print_results: 是否打印结果
    
    返回:
        dict: 包含所有指标的字典
    """
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if thresholds is not None:
        y_pred = apply_thresholds(y_prob, thresholds)
    else:
        y_pred = (y_prob > 0.5).astype(int)
    
    # 1. Micro-F1: 全局计算TP, FP, FN
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    # 2. Macro-F1: 每个标签单独计算F1，然后平均
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 3. Hamming Loss: 错误预测的标签比例
    hamming_loss_score = hamming_loss(y_true, y_pred)
    
    # 4. Subset Accuracy: 完全匹配的样本比例
    subset_accuracy = np.mean(np.all(y_true == y_pred, axis=1))
    
    # 5. AUC: 每个标签的AUC平均（macro-average）
    try:
        # 检查每个标签是否都有正负样本
        auc_scores = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:  # 确保有正负样本
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                auc_scores.append(auc)
            else:
                # 如果某个标签只有一个类别，跳过或设为NaN
                print(f"Warning: Label {i} has only one class, skipping AUC calculation")
        
        if auc_scores:
            macro_auc = np.mean(auc_scores)
        else:
            macro_auc = float('nan')
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        macro_auc = float('nan')
    
    results = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'hamming_loss': hamming_loss_score,
        'subset_accuracy': subset_accuracy,
        'macro_auc': macro_auc
    }
    
    if print_results:
        print("\n" + "="*50)
        if obj_info is not None:
            print(obj_info)
        else:
            print("多标签分类评估结果")
        print("="*50)
        print(f"Micro-F1:        {micro_f1:.4f}")
        print(f"Macro-F1:        {macro_f1:.4f}")
        print(f"Hamming Loss:    {hamming_loss_score:.4f}")
        print(f"Subset Accuracy: {subset_accuracy:.4f}")
        if not np.isnan(macro_auc):
            print(f"Macro-AUC:       {macro_auc:.4f}")
        else:
            print(f"Macro-AUC:       N/A (calculation failed)")
        print("="*50)
    
    return results

def get_dataset(n_total_samples=3000, test_split=0.2):
    dataset = openml.datasets.get_dataset(41471)
    X, _, _, _ = dataset.get_data(dataset_format='dataframe')
    X = X.sample(frac=1).reset_index(drop=True).head(n_total_samples)
    cols = X.columns[-6:]
    y = X[cols].map(lambda v: 1 if v else 0)
    X = X.drop(columns=cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=test_split, random_state=42
    )
    return X_train, X_test, y_train, y_test

def main():
    config = {
        'd_model': 32,
        'hidden': 8,
        'n_heads': 4,
        'epochs': 30,
        'batch_size': 128
    }

    X_train, X_test, y_train, y_test = get_dataset()
    n_labels = y_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPWithAttention(
        n_labels=n_labels,
        d_model=config['d_model'],
        hidden=config['hidden'],
        n_heads=config['n_heads']
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    tabpfn_probs, target = gen_oof_data(X_train, y_train)
    train(
        model=model,
        X_train=tabpfn_probs,
        y_train=target,
        optimizer=optimizer,
        criterion=criterion,
        save_model=False,
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    y_test_prob = predict(model, X_train, y_train, X_test)
    
    # 评估模型性能
    evaluate_multilabel(y_test, y_test_prob)
    print(config)
    
    
if __name__ == '__main__':
    main()
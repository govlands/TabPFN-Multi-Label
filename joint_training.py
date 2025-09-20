from functools import partial
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score, log_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from feature_label_attn import JointFeatureLabelAttn
from mlp_att import evaluate_multilabel, get_dataset


class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_tabpfns(n_labels, device, pfn_n_estimators, random_seed=42) -> list[TabPFNClassifier]:
    """Create m TabPFNClassifier instances (binary per-label)."""
    pfn_cfg = dict(
        ignore_pretraining_limits=True,
        device=device,
        n_estimators=pfn_n_estimators,
        random_state=random_seed,
        inference_precision=torch.float32,
        fit_mode="batched",
        differentiable_input=True,
    )
    tabpfns: list[TabPFNClassifier] = []
    for _ in range(n_labels):
        clf = TabPFNClassifier(**pfn_cfg)
        clf._initialize_model_variables()
        tabpfns.append(clf)
    return tabpfns


def make_shared_split_fn(train_idx: np.ndarray, test_idx: np.ndarray, prefer_torch: bool = False):
    """Return a split_fn that applies the same index split to any (X,y).

    If prefer_torch is True, the returned split_fn yields torch.Tensor slices
    (suitable for differentiable preprocessors). Otherwise it yields numpy
    arrays (suitable for non-differentiable code paths).
    """
    def split_fn(X, y, random_state=None, **kwargs):
        is_torch_input = torch.is_tensor(X) or torch.is_tensor(y)
        # If caller prefers torch outputs and we can produce them, return tensors.
        if prefer_torch:
            # If inputs are numpy, convert selected slices to torch
            if is_torch_input:
                X_tr = X[train_idx]
                X_te = X[test_idx]
                y_tr = y[train_idx]
                y_te = y[test_idx]
            else:
                X_tr = torch.from_numpy(X[train_idx])
                X_te = torch.from_numpy(X[test_idx])
                y_tr = torch.from_numpy(y[train_idx])
                y_te = torch.from_numpy(y[test_idx])
            return X_tr, X_te, y_tr, y_te

        # Default: return numpy arrays (backwards-compatible)
        if is_torch_input:
            X_tr = X[train_idx].cpu().numpy()
            X_te = X[test_idx].cpu().numpy()
            y_tr = y[train_idx].cpu().numpy()
            y_te = y[test_idx].cpu().numpy()
        else:
            X_tr = X[train_idx]
            X_te = X[test_idx]
            y_tr = y[train_idx]
            y_te = y[test_idx]
        return X_tr, X_te, y_tr, y_te
    return split_fn


def save_model_e2e(tabpfns, joint_model, optimizer, config, n_features, filepath):
    """
    保存端到端联合训练的完整模型状态
    
    Args:
        tabpfns: List of TabPFNClassifier instances
        joint_model: JointFeatureLabelAttn model
        optimizer: AdamW optimizer
        n_features: Number of input features
        filepath: Path to save the model (without extension)
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not filepath.endswith(('.pt', '.pth')):
        filepath = f"{filepath}_{timestamp}.pt"
    
    # 收集所有TabPFN模型的状态
    tabpfn_states = []
    for i, tabpfn in enumerate(tabpfns):
        # 保存TabPFN的关键组件
        tabpfn_state = {
            'model_state_dict': tabpfn.model_.state_dict(),
            'config': getattr(tabpfn, 'config_', None),
            'device': str(tabpfn.device) if hasattr(tabpfn, 'device') else None,
            'index': i
        }
        tabpfn_states.append(tabpfn_state)
    
    # 完整的保存状态
    save_state = {
        'joint_model_state_dict': joint_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tabpfn_states': tabpfn_states,
        'config': config,
        'n_features': n_features,
        'n_labels': len(tabpfns),
        'timestamp': timestamp,
        'model_type': 'e2e_joint_training'
    }
    
    torch.save(save_state, filepath)
    print(f"✅ End-to-end model saved to: {filepath}")
    
    # 同时保存配置文件
    config_path = filepath.replace('.pt', '_config.txt')
    with open(config_path, 'w') as f:
        f.write("=== End-to-End Joint Training Configuration ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of features: {n_features}\n")
        f.write(f"Number of labels: {len(tabpfns)}\n")
        f.write(f"Joint model: {joint_model.__class__.__name__}\n")
        f.write("\nTraining Config:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nOptimizer: {optimizer.__class__.__name__}\n")
        f.write("Model components:\n")
        f.write(f"  - TabPFN classifiers: {len(tabpfns)}\n")
        f.write(f"  - Joint attention model: {joint_model.__class__.__name__}\n")
    
    return filepath


def load_model_e2e(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # 加载保存的状态
    save_state = torch.load(filepath, map_location=device, weights_only=False)
    
    if save_state.get('model_type') != 'e2e_joint_training':
        print("⚠️  Warning: This doesn't appear to be an e2e joint training model")
    
    config = save_state['config']
    tabpfn_config = config['tabpfn_config']
    n_features = save_state['n_features']
    n_labels = save_state['n_labels']
    
    print(f"📦 Loading end-to-end model from: {filepath}")
    print(f"   Timestamp: {save_state.get('timestamp', 'unknown')}")
    print(f"   Features: {n_features}, Labels: {n_labels}")
    
    tabpfns = [None] * n_labels
    tabpfn_states = save_state['tabpfn_states']
    for state in tabpfn_states:
        idx = state['index']
        tabpfn = TabPFNClassifier(**tabpfn_config)
        tabpfn._initialize_model_variables()
        try:
            tabpfn.model_.load_state_dict(state['model_state_dict'])
            print(f"   ✅ Restored TabPFN {idx} model state")
        except Exception as e:
            print(f"   ⚠️  Could not restore TabPFN {idx} state: {e}")
        tabpfns[idx] = tabpfn
    
    # 重建联合注意力模型
    joint_model = JointFeatureLabelAttn(n_features=n_features, n_labels=n_labels).to(device)
    
    # 恢复模型状态
    joint_model.load_state_dict(save_state['joint_model_state_dict'])
    
    # 重建优化器
    # 分离参数组：joint模型参数 vs TabPFN参数
    joint_params = list(joint_model.parameters())
    pfn_params = []
    for tabpfn in tabpfns:
        pfn_params.extend(list(tabpfn.model_.parameters()))
    
    optimizer = AdamW([
        {'params': joint_params, 'lr': config.get('joint_lr', 1e-4)},
        {'params': pfn_params, 'lr': config.get('pfn_lr', 1e-5)}
    ])
    
    # 恢复优化器状态
    try:
        optimizer.load_state_dict(save_state['optimizer_state_dict'])
        print("   ✅ Restored optimizer state")
    except Exception as e:
        print(f"   ⚠️  Could not restore optimizer state: {e}")
        print("   🔧 Using fresh optimizer state")
    
    print(f"✅ End-to-end model loaded successfully")
    
    return tabpfns, joint_model, optimizer, config


def predict_e2e(tabpfns, joint_model, X_train, y_train, X_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    端到端预测函数：使用TabPFN分类器和联合注意力模型进行预测
    
    Args:
        tabpfns: List of TabPFNClassifier instances
        joint_model: JointFeatureLabelAttn model  
        X_train: Training features (np.ndarray)
        y_train: Training labels (np.ndarray)
        X_test: Test features (np.ndarray)
        device: Device for computation
    
    Returns:
        np.ndarray: Predicted probabilities for test samples (n_test_samples, n_labels)
    """
    
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    
    n_test = X_test.shape[0]
    n_labels = y_train.shape[1]
    joint_model = joint_model.to(device)
    
    joint_model.eval()
    z_logits_list = []
    with torch.no_grad():
        for label_idx in range(n_labels):
            y_label = y_train[:, label_idx]
            eval_cfg = {"device": device, "fit_mode": "fit_preprocessors"}
            eval_clf = clone_model_for_evaluation(tabpfns[label_idx], eval_cfg, TabPFNClassifier)
            eval_clf.fit(X_train, y_label)
            logits = eval_clf.predict_logits(X_test)
            logits = logits[:, 1]
            z_logits_list.append(logits)
    
    Z_logits = np.stack(z_logits_list, axis=1)
    
    X_test_tensor = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
    Z_logits_tensor = torch.from_numpy(Z_logits).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        joint_logits = joint_model(X_test_tensor, Z_logits_tensor)
        joint_probs = torch.sigmoid(joint_logits)
        predictions = joint_probs.cpu().numpy()
    
    return predictions


def predict_e2e_batch(tabpfns, joint_model, X_train, y_train, X_test, batch_size=1024, pred_time=4):

    # Coerce to numpy for sampling convenience
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)

    n_train = X_train.shape[0]
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # If batch_size larger than available training samples, sample with replacement
    replace = batch_size > n_train

    preds_accum = None
    for t in range(pred_time):
        # sample indices
        idx = np.random.choice(n_train, size=min(batch_size, n_train), replace=replace)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
        # run full pipeline for this subsample
        preds = predict_e2e(tabpfns, joint_model, X_sub, y_sub, X_test)
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if preds_accum is None:
            preds_accum = preds.astype(np.float64)
        else:
            preds_accum += preds.astype(np.float64)

    preds_mean = preds_accum / float(pred_time)
    return preds_mean


def train_e2e(
    n_labels,
    X_train,
    y_train,
    joint_model,
    tabpfns,
    bce_loss,
    optim,
    batch_size=512,
    epochs=10,
    n_splits=4,
    random_seed=42,
    device='cuda',
    lambda1=0.5,
    lambda2=0,
    validation_split=0.15,
    early_stopping_patience=5,
    early_stopping_delta=1e-4
) -> None:
    """End-to-end joint training: m TabPFNs + JointFeatureLabelAttn with combined loss with early stopping."""
    from sklearn.model_selection import train_test_split
    
    # Split training data into train and validation sets
    if validation_split > 0:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, 
            test_size=validation_split, 
            random_state=random_seed,
            stratify=None  # Multi-label stratification is complex, so we skip it for now
        )
        print(f"--- Data Split: Train={X_train_split.shape[0]}, Val={X_val_split.shape[0]} ---")
    else:
        X_train_split, y_train_split = X_train, y_train
        X_val_split, y_val_split = None, None
        print("--- No validation split, using full training data ---")
    
    # DataLoader over training data
    train_loader = DataLoader(
        TabDataset(X_train_split, y_train_split),
        batch_size=batch_size, shuffle=True
    )
    
    # Validation DataLoader if validation split exists
    if X_val_split is not None:
        val_loader = DataLoader(
            TabDataset(X_val_split, y_val_split),
            batch_size=batch_size, shuffle=False
        )
    else:
        val_loader = None
    
    joint = joint_model.to(device)
    bce_logits = bce_loss
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    DEBUG = False

    print("--- Starting End-to-End Joint Training ---")
    for epoch in range(1, epochs + 1):
        joint.train()
        running_train_loss = 0.0
        train_steps = 0
        
        # Training phase
        for X_mb_t, Y_mb_t in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            X_mb = X_mb_t.numpy()   # mb: mini-batch
            Y_mb = Y_mb_t.numpy().astype(np.int64)
            
            n_b = X_mb.shape[0]
            if DEBUG: print(f"shape X_mb: {X_mb.shape}, shape Y_mb: {Y_mb.shape}")
            # shared split indices within batch
            idx = np.arange(n_b)
            
            kf_iter = MultilabelStratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            ).split(idx, Y_mb)
            
            batch_loss_sum = None
            n_used_folds = 0
            
            for train_idx, test_idx in kf_iter:
                if DEBUG: print(f"n_train: {len(train_idx)}, n_test: {len(test_idx)}")
                skip_fold = False
                for i in range(n_labels):
                    if len(np.unique(Y_mb[test_idx, i])) < 2 or len(np.unique(Y_mb[train_idx, i])) < 2:
                        skip_fold = True
                        break
                if skip_fold: continue

                split_fn = make_shared_split_fn(train_idx, test_idx, prefer_torch=True)

                # Build Z logits for test subset across labels
                n_te = len(test_idx)
                pos_list = []  # 保存每个标签的 pos_logits（保持计算图）

                # For each label, get preprocessed data and run PFN fit+forward
                for ell in range(n_labels):
                    y_ell_t = Y_mb_t[:, ell]
                    ds = tabpfns[ell].get_preprocessed_datasets(
                        X_mb_t, y_ell_t, split_fn, max_data_size=None
                    )
                    dl = DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator)
                    for (
                        X_tr_list,
                        X_te_list,
                        y_tr_list,
                        y_te_raw,  # unused here
                        cat_ixs,
                        confs,
                    ) in dl:
                        if DEBUG:
                            print(f"len(X_tr_list): {len(X_tr_list)}")
                            print(f"shape X_tr_list[0]: {X_tr_list[0].shape}")
                            print(f"len(y_tr_list): {len(y_tr_list)}")
                            print(f"shape y_tr_list[0]: {y_tr_list[0].shape}")
                            
                        # Fit on context
                        tabpfns[ell].fit_from_preprocessed(X_tr_list, y_tr_list, cat_ixs, confs, no_refit=True)
                        # Forward logits: (batch=1, classes(2), n_te)
                        preds = tabpfns[ell].forward(X_te_list, return_logits=True)
                        
                        if DEBUG:
                            print(f"shape preds: {preds.shape}")
                        
                        if isinstance(preds, (list, tuple)):
                            preds = preds[0]
                        # squeeze batch dim
                        if preds.dim() == 3 and preds.shape[0] == 1:
                            preds = preds.squeeze(0)
                        # 选取正类 logit -> 得到 1D 向量 (n_te,)
                        if preds.shape[0] == 2 and preds.shape[1] == n_te:
                            pos_logits = preds[1, :]  # shape (n_te,)
                        elif preds.shape[1] == 2 and preds.shape[0] == n_te:
                            pos_logits = preds[:, 1]  # shape (n_te,)
                        else:
                            raise RuntimeError(f"Unexpected PFN logits shape for label {ell}: {tuple(preds.shape)}")

                        # 规范 device/dtype/shape 并保留计算图
                        pos = pos_logits.to(device=device, dtype=torch.float32).contiguous().view(-1)
                        if DEBUG: print(f"shape pos: {pos.shape}")
                        
                        if pos.shape[0] != n_te:
                            raise RuntimeError(f"Label {ell}: expected {n_te} logits, got {pos.shape[0]}")
                        pos_list.append(pos)
                        break  # only one item

                # 将各标签 logits 堆叠成 (n_te, n_labels)，此处保留了从 pos_logits 到 PFN 参数的计算图
                Z_te = torch.stack(pos_list, dim=1)  # shape (n_te, n_labels)
                if DEBUG: print(f"shape Z_te: {Z_te.shape}")

                # Joint forward on raw features of test subset
                X_te_raw = torch.from_numpy(X_mb[test_idx]).to(device=device, dtype=torch.float32)
                Y_te = torch.from_numpy(Y_mb[test_idx]).to(device=device, dtype=torch.float32)
                y2_logits = joint(X_te_raw, Z_te)
                if DEBUG: print(f"shape y2_logits: {y2_logits.shape}")

                # Combined loss
                loss2 = bce_logits(y2_logits, Y_te)
                loss_z = bce_logits(Z_te, Y_te)
                # KL between sigmoid(Z) and stopgrad(sigmoid(y2_logits))
                if lambda2 > 0:
                    eps = 1e-6
                    p = torch.sigmoid(Z_te).clamp(eps, 1 - eps)
                    q = torch.sigmoid(y2_logits).detach().clamp(eps, 1 - eps)
                    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
                    loss_kl = kl.mean()
                else:
                    loss_kl = torch.tensor(0.0, device=device)
                loss = loss2 + lambda1 * loss_z + lambda2 * loss_kl

                batch_loss_sum = loss if batch_loss_sum is None else (batch_loss_sum + loss)
                n_used_folds += 1

            if n_used_folds > 0:
                loss_mean = batch_loss_sum / n_used_folds
                optim.zero_grad()
                loss_mean.backward()
                optim.step()
                running_train_loss += float(loss_mean.item())
                train_steps += 1

        # Calculate average training loss
        avg_train_loss = running_train_loss / max(1, train_steps)
        
        # Validation phase
        val_loss = None
        if val_loader is not None:
            joint.eval()
            running_val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for X_mb_t, Y_mb_t in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                    X_mb = X_mb_t.numpy()
                    Y_mb = Y_mb_t.numpy().astype(np.int64)
                    
                    n_b = X_mb.shape[0]
                    idx = np.arange(n_b)
                    
                    kf_iter = MultilabelStratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=random_seed
                    ).split(idx, Y_mb)
                    
                    batch_loss_sum = None
                    n_used_folds = 0
                    
                    for train_idx, test_idx in kf_iter:
                        skip_fold = False
                        for i in range(n_labels):
                            if len(np.unique(Y_mb[test_idx, i])) < 2 or len(np.unique(Y_mb[train_idx, i])) < 2:
                                skip_fold = True
                                break
                        if skip_fold: continue

                        split_fn = make_shared_split_fn(train_idx, test_idx, prefer_torch=True)
                        n_te = len(test_idx)
                        pos_list = []

                        # Validation forward pass (similar to training but without gradients)
                        for ell in range(n_labels):
                            y_ell_t = Y_mb_t[:, ell]
                            ds = tabpfns[ell].get_preprocessed_datasets(
                                X_mb_t, y_ell_t, split_fn, max_data_size=None
                            )
                            dl = DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator)
                            for (
                                X_tr_list,
                                X_te_list,
                                y_tr_list,
                                y_te_raw,
                                cat_ixs,
                                confs,
                            ) in dl:
                                tabpfns[ell].fit_from_preprocessed(X_tr_list, y_tr_list, cat_ixs, confs, no_refit=True)
                                preds = tabpfns[ell].forward(X_te_list, return_logits=True)
                                
                                if isinstance(preds, (list, tuple)):
                                    preds = preds[0]
                                if preds.dim() == 3 and preds.shape[0] == 1:
                                    preds = preds.squeeze(0)
                                    
                                if preds.shape[0] == 2 and preds.shape[1] == n_te:
                                    pos_logits = preds[1, :]
                                elif preds.shape[1] == 2 and preds.shape[0] == n_te:
                                    pos_logits = preds[:, 1]
                                else:
                                    raise RuntimeError(f"Unexpected PFN logits shape for label {ell}: {tuple(preds.shape)}")

                                pos = pos_logits.to(device=device, dtype=torch.float32).contiguous().view(-1)
                                if pos.shape[0] != n_te:
                                    raise RuntimeError(f"Label {ell}: expected {n_te} logits, got {pos.shape[0]}")
                                pos_list.append(pos)
                                break

                        Z_te = torch.stack(pos_list, dim=1)
                        X_te_raw = torch.from_numpy(X_mb[test_idx]).to(device=device, dtype=torch.float32)
                        Y_te = torch.from_numpy(Y_mb[test_idx]).to(device=device, dtype=torch.float32)
                        y2_logits = joint(X_te_raw, Z_te)

                        # Calculate validation loss (same as training loss)
                        loss2 = bce_logits(y2_logits, Y_te)
                        loss_z = bce_logits(Z_te, Y_te)
                        if lambda2 > 0:
                            eps = 1e-6
                            p = torch.sigmoid(Z_te).clamp(eps, 1 - eps)
                            q = torch.sigmoid(y2_logits).detach().clamp(eps, 1 - eps)
                            kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
                            loss_kl = kl.mean()
                        else:
                            loss_kl = torch.tensor(0.0, device=device)
                        loss = loss2 + lambda1 * loss_z + lambda2 * loss_kl

                        batch_loss_sum = loss if batch_loss_sum is None else (batch_loss_sum + loss)
                        n_used_folds += 1

                    if n_used_folds > 0:
                        loss_mean = batch_loss_sum / n_used_folds
                        running_val_loss += float(loss_mean.item())
                        val_steps += 1

            val_loss = running_val_loss / max(1, val_steps)

        # Print epoch results
        if val_loss is not None:
            print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}, val_loss = {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - early_stopping_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                # Save best model state
                best_model_state = {
                    'joint_state_dict': joint.state_dict().copy(),
                    'tabpfn_states': [clf.model_.state_dict().copy() for clf in tabpfns],
                    'optimizer_state_dict': optim.state_dict().copy(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss
                }
                print(f"    ✅ New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"    ⏳ No improvement for {patience_counter} epochs")
                
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    print(f"📊 Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                    break
        else:
            print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}")
    
    # Restore best model if early stopping occurred and we have a saved state
    if best_model_state is not None:
        print(f"🔄 Restoring best model from epoch {best_epoch}")
        joint.load_state_dict(best_model_state['joint_state_dict'])
        for i, clf in enumerate(tabpfns):
            clf.model_.load_state_dict(best_model_state['tabpfn_states'][i])
        optim.load_state_dict(best_model_state['optimizer_state_dict'])
        print(f"✅ Best model restored (val_loss: {best_val_loss:.4f})")

    print("--- ✅ End-to-End Joint Training Finished ---")


class MultiLabelTabPFN_e2eFeatureLabel:
    def __init__(
        self,
        n_features,
        n_labels,
        pfn_n_estimators=4,
        random_seed=42,
        batch_size=512,
        epochs=10,
        n_splits=4,
        lambda1=0.5,
        lambda2=0,
        tabpfn_lr=1e-5,
        joint_lr=3e-3,
        validation_split=0.15,
        early_stopping_patience=5,
        early_stopping_delta=1e-4,
    ):
        self.n_features = n_features
        self.n_labels = n_labels
        self.pfn_n_estimators = pfn_n_estimators
        self.seed = random_seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_splits = n_splits
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tabpfn_lr = tabpfn_lr
        self.joint_lr = joint_lr
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tabpfns = build_tabpfns(
            n_labels=self.n_labels,
            device=self.device,
            pfn_n_estimators=self.pfn_n_estimators,
            random_seed=self.seed
        )
        self.joint = JointFeatureLabelAttn(
            n_features=self.n_features,
            n_labels=self.n_labels
        ).to(self.device)
        
        pfn_params = []
        for clf in self.tabpfns:
            pfn_params += list(clf.model_.parameters())
        self.optim = AdamW([
            {"params": self.joint.parameters(), "lr": self.joint_lr},
            {"params": pfn_params, "lr": self.tabpfn_lr},
        ])
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
    
    
    def fit(
        self,
        X_train,
        y_train,
    ):
        train_e2e(
            n_labels=self.n_labels,
            X_train=X_train,
            y_train=y_train,
            joint_model=self.joint,
            tabpfns=self.tabpfns,
            bce_loss=self.bce_loss,
            optim=self.optim,
            batch_size=self.batch_size,
            epochs=self.epochs,
            n_splits=int(self.n_splits),
            random_seed=self.seed,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            device=str(self.device),
            validation_split=self.validation_split,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_delta=self.early_stopping_delta,
        )
        self.X_train = X_train
        self.y_train = y_train
    
    
    def predict_proba(
        self,
        X_test,
        mode='batch'
    ):
        if mode == 'batch':
            # n_splits = int(self.n_splits)
            # batch_size = int(self.batch_size * (1 - 1 / n_splits))
            batch_size = self.batch_size
            return predict_e2e_batch(
                tabpfns=self.tabpfns,
                joint_model=self.joint,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=X_test,
                batch_size=batch_size,
                pred_time=5,
            )
        elif mode == 'all':
            return predict_e2e(
                tabpfns=self.tabpfns,
                joint_model=self.joint,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=X_test,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Valid modes are 'batch' or 'all'.")
    
    
    def save(self):
        pfn_cfg = dict(
            ignore_pretraining_limits=True,
            device=self.device,
            n_estimators=self.pfn_n_estimators,
            random_state=self.seed,
            inference_precision=torch.float32,
            fit_mode="batched",
            differentiable_input=True,
        )

        config = {
            "tabpfn_config": pfn_cfg,
            "joint_lr": self.joint_lr,
            "pfn_lr": self.tabpfn_lr,
            "n_features": self.n_features,
            "n_labels": self.n_labels,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_splits": self.n_splits,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "random_seed": self.seed,
        }

        os.makedirs("models", exist_ok=True)
        filepath = os.path.join("models", "e2e_joint_model")
        return save_model_e2e(self.tabpfns, self.joint, self.optim, config, self.n_features, filepath)
    
    
    def load(self, path):
        tabpfns, joint_model, optimizer, config = load_model_e2e(filepath=path)
        self.tabpfns = tabpfns
        self.joint = joint_model
        self.optim = optimizer


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    X_train, X_test, y_train, y_test = get_dataset(use_synthetic_data=True, synthetic_size=1000)
    n_features, n_labels = X_train.shape[1], y_train.shape[1]
    
    # Create model with early stopping enabled
    model = MultiLabelTabPFN_e2eFeatureLabel(
        n_features=n_features,
        n_labels=n_labels,
        epochs=20,                      # Increased epochs since we have early stopping
        batch_size=400,
        n_splits=4,
        pfn_n_estimators=4,
        validation_split=0.15,          # Use 15% of training data for validation
        early_stopping_patience=5,      # Stop if no improvement for 5 epochs
        early_stopping_delta=1e-4,      # Minimum improvement threshold
    )
    
    print("🚀 Starting training with early stopping...")
    model.fit(X_train, y_train)
    
    print("\n📊 Making predictions...")
    pred_result_all = model.predict_proba(X_test, mode='all')
    pred_result_batch = model.predict_proba(X_test, mode='batch')

    print("\n📈 Evaluation Results:")
    print("=" * 50)
    print("pred_result_all performance:")
    evaluate_multilabel(y_test, pred_result_all)
    print("\npred_result_batch performance:")
    evaluate_multilabel(y_test, pred_result_batch)
    
    # save_path = model.save()
    # test_load = MultiLabelTabPFN_e2eFeatureLabel(
    #     n_features=n_features,
    #     n_labels=n_labels,
    #     epochs=5,
    # )
    # test_load.load(save_path)


if __name__ == "__main__":
    main()
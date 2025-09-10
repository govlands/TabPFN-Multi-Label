import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mlp_att import evaluate_multilabel, train, predict

def formalize_output_probas(probas, n_labels):
    """
    把 predict_proba 的输出统一成 (n_samples, n_labels) 的正类概率矩阵。
    兼容返回 list/array/3D array 等情况。
    """
    # list of arrays (one per label)
    if isinstance(probas, list):
        cols = []
        for p in probas:
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] > 1:
                cols.append(p[:, 1])
            else:
                cols.append(p.ravel())
        return np.column_stack(cols)

    probas = np.asarray(probas)
    # possible shape (n_labels, n_samples, n_classes)
    if probas.ndim == 3:
        # bring to (n_samples, n_labels)
        n_labels0 = probas.shape[0]
        cols = []
        for i in range(n_labels0):
            if probas.shape[2] > 1:
                cols.append(probas[i, :, 1])
            else:
                cols.append(probas[i, :, 0])
        return np.column_stack(cols)
    # possible shape (n_samples, n_labels) or (n_samples, n_labels*classes) - assume already (n_samples, n_labels)
    if probas.ndim == 2 and probas.shape[1] == n_labels:
        return probas
    # fallback: raise
    raise ValueError("无法识别 predict_proba 输出格式，请检查 base estimator 是否支持 predict_proba。")

class MLPBaseline(nn.Module):
    def __init__(self, n_features, n_labels, hidden=(64, 32), p_drop=0.2, use_layernorm=True):
        super().__init__()
        layers = []
        d_prev = n_features
        for d_h in hidden:
            layers.append(nn.Linear(d_prev, d_h))
            layers.append(nn.LayerNorm(d_h) if use_layernorm else nn.BatchNorm1d(d_h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p_drop))
            d_prev = d_h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d_prev, n_labels)
        
    def forward(self, X):
        # X:(batch, n_features)
        z = self.backbone(X)
        return self.head(z)

def main():
    # 生成示例多标签数据
    X, Y = make_multilabel_classification(n_samples=200, n_features=20, n_classes=6,
                                          n_labels=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    n_labels = y_train.shape[1]
    n_features = X_train.shape[1]
    
    model_dict = {
        'Logistic': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            max_features='sqrt',
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42
        ),
        'TabPFN': TabPFNClassifier(),
    }
    
    BR_results = {}
    CC_results = {}
    mlp_results = {}
    for name, model in model_dict.items():
        br_model = MultiOutputClassifier(model, n_jobs=-1)
        br_model.fit(X_train, y_train)
        if hasattr(br_model, "predict_proba"):
            probas = br_model.predict_proba(X_test)
            probas_mat = formalize_output_probas(probas, n_labels)
        else:
            probas_mat = br_model.predict(X_test).astype(float)
        BR_results[name] = evaluate_multilabel(y_test, probas_mat)
        
    for name, model in model_dict.items():
        chain = ClassifierChain(model, order='random', random_state=42)
        chain.fit(X_train, y_train)
        if hasattr(chain, "predict_proba"):
            try:
                probas_chain = chain.predict_proba(X_test)
                probas_chain_mat = formalize_output_probas(probas_chain, n_labels)
            except Exception:
                # fallback: 手工按顺序用 estimators_ 逐步构造概率或以硬预测作为近似
                X_aug = X_test.copy()
                cols = []
                for est in chain.estimators_:
                    p = est.predict_proba(X_aug)
                    if p.ndim == 2 and p.shape[1] > 1:
                        pos = p[:, 1]
                    else:
                        pos = p.ravel()
                    cols.append(pos)
                    # append hard prediction as next feature for chain (greedy)
                    X_aug = np.hstack([X_aug, (pos > 0.5).astype(int)[:, None]])
                probas_chain_mat = np.column_stack(cols)
        else:
            # 没有 predict_proba，退回到 predict 的 0/1
            probas_chain_mat = chain.predict(X_test).astype(float)
        CC_results[name] = evaluate_multilabel(y_test, probas_chain_mat)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPBaseline(
        n_features=n_features,
        n_labels=n_labels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    train(
        model = model,
        X_train=X_train,
        y_train=y_train,
        optimizer=optimizer,
        criterion=criterion,
        epochs=20,
        batch_size=128
    )
    
    model.eval()
    with torch.no_grad():
        X_t= torch.from_numpy(X_test).to(device)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        mlp_results['MLP'] = evaluate_multilabel(y_test, probs)

if __name__ == "__main__":
    main()
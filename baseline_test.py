import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
from mlp_att import evaluate_multilabel, get_dataset, MultiLabelTabPFN_LabelOnly, train, MLPWithAttention, predict, load_model
from feature_label_attn import MultiLabelTabPFN_FeatureLabel, predict_joint, JointFeatureLabelAttn
from joint_training import MultiLabelTabPFN_e2eFeatureLabel

from utils import formalize_output_probas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

def classes_list_from_chain(chain, n_labels):
    """
    返回长度为 n_labels 的列表，索引对应原始标签列，
    每项为该标签对应子分类器的 classes_（或者 None）。
    """
    classes_list = [None] * n_labels
    if not hasattr(chain, "estimators_"):
        return classes_list

    # chain.order_ 给出 estimators_ 对应的原始标签索引顺序
    order = getattr(chain, "order_", None)
    if order is None:
        # sklearn 旧版本可能没有 order_，假定按 0..n_labels-1
        order = np.arange(n_labels)

    for est, lbl_idx in zip(chain.estimators_, order):
        classes_list[int(lbl_idx)] = getattr(est, "classes_", None)
    return classes_list

class MLPBaseline(nn.Module):
    def __init__(self, n_features, n_labels, hidden=(256, 128), p_drop=0.2, use_layernorm=True):
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

def visualize_model_comparison(results_dict, save_prefix=None, figsize=(16, 10)):
    """
    对比多个模型的性能指标，分组显示并保存多个图像文件
    
    参数:
        results_dict: dict，格式 {model_name: {metric_name: value, ...}, ...}
        save_prefix: str，保存图像的前缀路径（可选）
        figsize: tuple，每组图像的大小
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 整理数据
    df = pd.DataFrame(results_dict).T
    df = df.fillna(0)  # 填充 NaN 值
    
    # 定义指标方向：True表示越大越好，False表示越小越好
    metric_directions = {
        'micro_f1': True,
        'macro_f1': True, 
        'subset_accuracy': True,
        'macro_auc': True,
        'hamming_loss': False  # hamming loss 越小越好
    }
    
    # 统一列名映射，兼容不同的命名约定
    column_mapping = {
        'hamming_loss': 'hamming',
        'subset_accuracy': 'subset_acc', 
        'macro_auc': 'auc'
    }
    
    # 重命名列以保持一致性
    df_renamed = df.rename(columns=column_mapping)
    
    # ==================== 图组 1: 各指标详细对比（按指标分组展示各模型表现） ====================
    # 目标：每个指标为一个组，组内并排显示不同模型的柱状值，便于比较同一指标下的模型表现
    # 将hamming loss分离到单独的图中，其他指标放在主图中
    
    # 主要指标（越高越好）
    main_metrics = [m for m in ['micro_f1', 'macro_f1', 'subset_acc', 'auc'] if m in df_renamed.columns]
    # hamming loss（越低越好）
    hamming_metrics = [m for m in ['hamming'] if m in df_renamed.columns]
    
    if len(main_metrics) == 0 and len(hamming_metrics) == 0:
        print("No metrics available for bar chart.")
    else:
        # 创建子图：第一个用于主要指标，第二个用于hamming loss
        if len(hamming_metrics) > 0:
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))
        else:
            fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None
        
        n_models = df_renamed.shape[0]
        colors = plt.cm.tab20(np.linspace(0, 1, n_models))
        
        # ======== 主要指标图 ========
        if len(main_metrics) > 0:
            df_main = df_renamed[main_metrics].copy()
            n_metrics = df_main.shape[1]
            x = np.arange(n_metrics)
            total_width = 0.8
            width = total_width / max(n_models, 1)
            
            for i, model_name in enumerate(df_main.index):
                values = df_main.loc[model_name].values
                ax1.bar(x + i * width - total_width/2 + width/2, values, width, label=model_name, color=colors[i])
                # annotate
                for xi, v in zip(x + i * width - total_width/2 + width/2, values):
                    ax1.text(xi, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in main_metrics])
            # 调整纵轴范围以增强视觉差异 - 主要指标通常在0.3-0.9之间
            main_values = df_main.values
            min_val = max(0, main_values.min() - 0.05)
            max_val = min(1.0, main_values.max() + 0.05)
            ax1.set_ylim(min_val, max_val)
            ax1.set_ylabel('分数（越高越好）')
            ax1.set_title('主要指标对比', fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(axis='y', alpha=0.3)
        
        # ======== Hamming Loss 图 ========
        if len(hamming_metrics) > 0 and ax2 is not None:
            df_hamming = df_renamed[hamming_metrics].copy()
            n_metrics_h = df_hamming.shape[1]
            x_h = np.arange(n_metrics_h)
            total_width_h = 0.8
            width_h = total_width_h / max(n_models, 1)
            
            for i, model_name in enumerate(df_hamming.index):
                values_h = df_hamming.loc[model_name].values
                ax2.bar(x_h + i * width_h - total_width_h/2 + width_h/2, values_h, width_h, label=model_name, color=colors[i])
                # annotate
                for xi, v in zip(x_h + i * width_h - total_width_h/2 + width_h/2, values_h):
                    ax2.text(xi, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax2.set_xticks(x_h)
            ax2.set_xticklabels([m.replace('_', ' ').title() for m in hamming_metrics])
            # 调整纵轴范围以增强视觉差异 - hamming loss通常在0.1-0.4之间
            hamming_values = df_hamming.values
            min_val_h = max(0, hamming_values.min() - 0.02)
            max_val_h = hamming_values.max() + 0.02
            ax2.set_ylim(min_val_h, max_val_h)
            ax2.set_ylabel('分数（越低越好）')
            ax2.set_title('Hamming Loss 对比', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_prefix:
            save_path1 = f"{save_prefix}_metrics_grouped.png"
            plt.savefig(save_path1, dpi=300, bbox_inches='tight')
            print(f"各指标对比柱状图已保存到: {save_path1}")
        plt.show()
    
    # ==================== 图组 2: 排名热力图 + 综合排名 ====================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize)
    
    # 2.1 排名热力图
    ranking_data = {}
    # 越大越好的指标
    for metric in ['micro_f1', 'macro_f1', 'subset_acc', 'auc']:
        if metric in df_renamed.columns:
            ranking_data[metric] = df_renamed[metric].rank(ascending=False, method='min').astype(int)
    
    # 越小越好的指标 (hamming loss)
    if 'hamming' in df_renamed.columns:
        ranking_data['hamming'] = df_renamed['hamming'].rank(ascending=True, method='min').astype(int)
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        
        sns.heatmap(ranking_df, annot=True, fmt='d', cmap='RdYlGn_r', 
                    ax=ax3, cbar_kws={'label': '排名'})
        ax3.set_title('各指标模型排名 (1=最佳)', fontweight='bold')
        ax3.set_xlabel('评估指标')
        ax3.set_ylabel('模型')
        
        # 2.2 综合排名
        avg_rank = ranking_df.mean(axis=1).sort_values()
        
        bars = ax4.barh(range(len(avg_rank)), avg_rank.values)
        ax4.set_yticks(range(len(avg_rank)))
        ax4.set_yticklabels(avg_rank.index)
        ax4.set_xlabel('平均排名')
        ax4.set_title('综合排名\n(平均排名越小越好)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签和颜色
        best_idx = avg_rank.index[0]
        worst_idx = avg_rank.index[-1]
        for i, (model, rank) in enumerate(avg_rank.items()):
            width = bars[i].get_width()
            ax4.text(width + 0.05, bars[i].get_y() + bars[i].get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
            
            if model == best_idx:
                bars[i].set_color('green')
                bars[i].set_alpha(0.7)
            elif model == worst_idx:
                bars[i].set_color('red')
                bars[i].set_alpha(0.7)
    
    plt.tight_layout()
    if save_prefix:
        save_path2 = f"{save_prefix}_ranking.png"
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"排名图已保存到: {save_path2}")
    plt.show()
    
    # ==================== 图组 3: 指标分布箱线图 ====================
    fig3, ax5 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]//2))
    
    # 准备箱线图数据
    plot_data = []
    plot_labels = []
    
    for metric in ['micro_f1', 'macro_f1', 'subset_acc', 'auc']:
        if metric in df_renamed.columns:
            plot_data.append(df_renamed[metric].values)
            plot_labels.append(metric.replace('_', ' ').title())
    
    if plot_data:
        bp = ax5.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        # 设置箱线图颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
            patch.set_facecolor(color)
        
        ax5.set_title('各指标数值分布', fontweight='bold')
        ax5.set_ylabel('分数')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_prefix:
        save_path3 = f"{save_prefix}_distribution.png"  
        plt.savefig(save_path3, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到: {save_path3}")
    plt.show()
    
    # 打印总结
    if 'ranking_df' in locals() and 'avg_rank' in locals():
        print("\n" + "="*60)
        print("模型性能总结")
        print("="*60)
        print("综合排名 (平均排名越小越好):")
        for i, (model, rank) in enumerate(avg_rank.items(), 1):
            print(f"{i:2d}. {model:<20} 平均排名: {rank:.2f}")
        
        print(f"\n🏆 最佳模型: {avg_rank.index[0]}")
        print(f"📊 性能详情:")
        best_model_metrics = df_renamed.loc[avg_rank.index[0]]
        for metric, value in best_model_metrics.items():
            if not np.isnan(value):
                direction = "↑" if metric_directions.get(metric, True) else "↓"
                print(f"   {metric}: {value:.4f} {direction}")
        
        return df_renamed, ranking_df, avg_rank
    else:
        return df_renamed, None, None

def main():
    # 生成示例多标签数据
    # X, Y = make_multilabel_classification(n_samples=200, n_features=20, n_classes=6,
    #                                       n_labels=2, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = get_dataset()
    n_labels = y_train.shape[1]
    n_features = X_train.shape[1]
    
    model_dict = {
        'Logistic': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            n_jobs=16,
            max_features='sqrt',
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=16,
            random_state=42,
            tree_method='hist',
            # device='cuda',
        ),
        'TabPFN': TabPFNClassifier(),
    }
    
    BR_results = {}
    CC_results = {}
    MLP_results = {}
    for name, model in model_dict.items():
        print(f"\n训练 BR+{name}...")
        br_model = MultiOutputClassifier(model, n_jobs=-1)
        br_model.fit(X_train, y_train)
        
        if hasattr(br_model, "predict_proba"):
            probas = br_model.predict_proba(X_test)
            # 传入每个子分类器的 classes_ 以便准确定位正类列
            classes_list = []
            if hasattr(br_model, "estimators_"):
                for est in br_model.estimators_:
                    classes_list.append(getattr(est, "classes_", None))
            probas_mat = formalize_output_probas(probas, n_labels, classes_list=classes_list)
        else:
            probas_mat = br_model.predict(X_test).astype(float)
        
        BR_results[name] = evaluate_multilabel(y_test, probas_mat, obj_info=f"BR+{name}")
        
    for name, model in model_dict.items():
        if name == 'TabPFN': continue
        print(f"\n训练 CC+{name}...")
        chain = ClassifierChain(model, order='random', random_state=42)
        chain.fit(X_train, y_train)
        
        if hasattr(chain, "predict_proba"):
            try:
                probas_chain = chain.predict_proba(X_test)
                # chain.estimators_ 顺序对应每个标签的子模型，传入 classes_ 有助于精确抽取
                classes_list = []
                if hasattr(chain, "estimators_"):
                    classes_list = classes_list_from_chain(chain, n_labels)
                probas_chain_mat = formalize_output_probas(probas_chain, n_labels, classes_list=classes_list)
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
        
        CC_results[name] = evaluate_multilabel(y_test, probas_chain_mat, obj_info=f"CC+{name}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n训练 MLP...")
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
        batch_size=64
    )
    
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()

        MLP_results['MLP'] = evaluate_multilabel(y_test, probs, obj_info=f"MLP")
        
    # TabPFN + Attention 模型（标签注意力）
    try:
        print(f"\n训练 TabPFN+Attention [labels only]...")
        # 使用封装的类
        model_label_only = MultiLabelTabPFN_LabelOnly(
            n_labels=n_labels,
            epochs=20,
            batch_size=128,
            early_stopping_patience=5,
            validation_split=0.2,
        )
        model_label_only.fit(X_train, y_train, save_model=True)
        probas = model_label_only.predict_proba(X_test)
        
        MLP_results['TabPFN+Attention'] = evaluate_multilabel(y_test, probas, obj_info="TabPFN+Attention [labels only]")
        print("✅ TabPFN+Attention [labels only] 模型已训练并保存")
    except Exception as e:
        print(f"❌ TabPFN+Attention [labels only] 训练失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        print(f"\n训练 TabPFN+Attention [features & labels]...")
        # 使用封装的类
        model_feature_label = MultiLabelTabPFN_FeatureLabel(
            n_features=n_features,
            n_labels=n_labels,
            epochs=20,
            batch_size=128,
            early_stopping_patience=5,
            validation_split=0.15,
        )
        model_feature_label.fit(X_train, y_train, save_model=True)
        probas = model_feature_label.predict_proba(X_test)
        
        MLP_results['TabPFN+Attention joint'] = evaluate_multilabel(y_test, probas, obj_info="TabPFN+Attention [features & labels]")
        print("✅ TabPFN+Attention [features & labels] 模型已训练并保存")
    except Exception as e:
        print(f"❌ TabPFN+Attention [features & labels] 训练失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        print(f"\n训练 TabPFN+Attention [e2e]...")
        # 使用封装的类 - 端到端训练
        model_e2e = MultiLabelTabPFN_e2eFeatureLabel(
            n_features=n_features,
            n_labels=n_labels,
            epochs=10,  # 较少的epochs防止显存不足
            batch_size=400,
            n_splits=4,  # 较少的splits减少内存占用
            pfn_n_estimators=4,  # 减少估计器数量
            validation_split=0.15,
            early_stopping_patience=5,
            early_stopping_delta=1e-4,
        )
        model_e2e.fit(X_train, y_train)
        probas = model_e2e.predict_proba(X_test, mode='all')
        
        # 保存模型
        save_path = model_e2e.save()
        print(f"✅ 端到端模型已保存到: {save_path}")
        
        MLP_results['TabPFN+Attention e2e'] = evaluate_multilabel(y_test, probas, obj_info="TabPFN+Attention [e2e]")
        print("✅ TabPFN+Attention [e2e] 模型已训练并保存")
    except Exception as e:
        print(f"❌ TabPFN+Attention [e2e] 训练失败: {e}")
        print("这可能是由于显存不足导致的，请考虑减少batch_size或使用CPU训练")
        import traceback
        traceback.print_exc()
    
    # 整合所有结果
    all_results = {}
    for name, result in BR_results.items():
        all_results[f'BR+{name}'] = result
    for name, result in CC_results.items():
        all_results[f'CC+{name}'] = result
    for name, result in MLP_results.items():
        all_results[name] = result
    
    # 可视化对比
    print("\n" + "="*80)
    print("开始可视化模型对比...")
    print("="*80)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # 确保输出目录存在
    import os
    os.makedirs('plots', exist_ok=True)
    
    visualize_model_comparison(
        all_results, 
        save_prefix=f'plots/model_comparison_{timestamp}',
        figsize=(16, 10)
    )

if __name__ == "__main__":
    main()
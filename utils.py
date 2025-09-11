import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Polygon
from datetime import datetime
import openml
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import os
import glob

def optimize_thresholds(y_true, y_prob, method='f1', search_points=100, verbose=False):
    """
    为每个标签优化阈值以最大化指定指标
    
    参数:
        y_true: 真实标签，形状 (n_samples, n_labels)
        y_prob: 预测概率，形状 (n_samples, n_labels)
        method: 优化目标，'f1' 或 'balanced_accuracy'
        search_points: 搜索点数量
        verbose: 是否打印详细信息
    
    返回:
        optimal_thresholds: 每个标签的最优阈值，形状 (n_labels,)
        threshold_scores: 每个标签在最优阈值下的得分
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n_labels = y_true.shape[1]
    
    optimal_thresholds = np.zeros(n_labels)
    threshold_scores = np.zeros(n_labels)
    
    # 为每个标签单独优化阈值
    for label_idx in range(n_labels):
        y_true_label = y_true[:, label_idx]
        y_prob_label = y_prob[:, label_idx]
        
        # 如果标签只有一个类别，跳过优化
        if len(np.unique(y_true_label)) <= 1:
            optimal_thresholds[label_idx] = 0.5
            threshold_scores[label_idx] = 0.0
            if verbose:
                print(f"Label {label_idx}: Only one class present, using default threshold 0.5")
            continue
        
        # 在概率范围内搜索最优阈值
        min_prob, max_prob = y_prob_label.min(), y_prob_label.max()
        thresholds = np.linspace(min_prob, max_prob, search_points)
        
        best_score = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_label = (y_prob_label >= threshold).astype(int)
            
            if method == 'f1':
                # 使用F1分数
                score = f1_score(y_true_label, y_pred_label, average='binary', zero_division=0)
            elif method == 'balanced_accuracy':
                # 使用平衡准确率
                from sklearn.metrics import balanced_accuracy_score
                score = balanced_accuracy_score(y_true_label, y_pred_label)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[label_idx] = best_threshold
        threshold_scores[label_idx] = best_score
        
        if verbose:
            print(f"Label {label_idx}: Optimal threshold = {best_threshold:.3f}, {method} = {best_score:.3f}")
    
    return optimal_thresholds, threshold_scores

def apply_thresholds(y_prob, thresholds):
    """
    使用给定阈值将概率转换为预测标签
    
    参数:
        y_prob: 预测概率，形状 (n_samples, n_labels)
        thresholds: 阈值数组，形状 (n_labels,)
    
    返回:
        y_pred: 预测标签，形状 (n_samples, n_labels)
    """
    y_prob = np.array(y_prob)
    thresholds = np.array(thresholds)
    
    # 广播比较
    y_pred = (y_prob >= thresholds[np.newaxis, :]).astype(int)
    return y_pred

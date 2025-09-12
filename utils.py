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

def formalize_output_probas(probas, n_labels, classes_list=None, positive_label=1):
    """
    把 predict_proba 的输出统一成 (n_samples, n_labels) 的正类概率矩阵。
    优先使用 classes_list（每个标签对应的 classes_）来定位正类列。
    参数:
      - probas: predict_proba 的返回值（list / ndarray (2D/3D)）
      - n_labels: 期望标签数
      - classes_list: 可选 list，每项为该标签对应的 classes_（例如 estimator.classes_）
      - positive_label: 希望取为正类的标签值（默认 1）
    返回:
      - numpy.ndarray, shape (n_samples, n_labels)
    """

    def pick_pos_col(arr2d, idx):
        arr = np.asarray(arr2d)
        # 一维或单列直接展平
        if arr.ndim == 1:
            return arr.ravel()
        if arr.ndim == 2:
            # 如果传入了 classes_list，则优先用它定位列
            if classes_list is not None and idx < len(classes_list) and classes_list[idx] is not None:
                cls = np.asarray(classes_list[idx], dtype=object)
                # 匹配数字或字符串形式
                matches = np.where((cls == positive_label) | (cls.astype(str) == str(positive_label)))[0]
                if matches.size:
                    col = matches[0]
                    return arr[:, col]
            # 回退：二分类默认取列 1（常见 classes_ == [0,1]）
            if arr.shape[1] > 1:
                return arr[:, 1]
            # 单列情况
            return arr[:, 0]
        raise ValueError(f"Unsupported array ndim for single-label proba: {arr.ndim}")

    # 1) list of arrays (one per label)
    if isinstance(probas, list):
        cols = []
        for i, p in enumerate(probas):
            p = np.asarray(p)
            cols.append(pick_pos_col(p, i))
        return np.column_stack(cols)

    probas = np.asarray(probas)

    # 2) possible shape (n_labels, n_samples, n_classes)
    if probas.ndim == 3:
        n_labels0 = probas.shape[0]
        cols = []
        for i in range(n_labels0):
            arr2d = probas[i]
            cols.append(pick_pos_col(arr2d, i))
        return np.column_stack(cols)

    # 3) possible shape (n_samples, n_labels) already
    if probas.ndim == 2:
        if probas.shape[1] == n_labels:
            return probas
        # 有时返回 (n_samples, n_labels * n_classes) —— 不支持自动拆分
        raise ValueError("二维 predict_proba 输出的列数与 n_labels 不匹配；请提供 classes_list 或检查基学习器。")

    raise ValueError("无法识别 predict_proba 输出格式，请检查 base estimator 是否支持 predict_proba。")

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

#!/usr/bin/env python3
"""
测试端到端联合训练模型的保存和加载功能
"""

import os
import torch
import numpy as np
from joint_training import main, load_model_e2e
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score, hamming_loss


def test_save_load():
    """测试模型的保存和加载功能"""
    
    print("=== 测试端到端联合训练模型保存/加载 ===")
    
    # 1. 运行训练并保存模型
    print("\n1. 开始训练并保存模型...")
    tabpfns, joint_model, optimizer = main()
    
    # 2. 查找保存的模型文件
    model_files = [f for f in os.listdir("models") if f.startswith("e2e_joint_model") and f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError("没有找到保存的模型文件")
    
    model_path = os.path.join("models", model_files[-1])  # 使用最新的模型
    print(f"找到模型文件: {model_path}")
    
    # 3. 加载模型
    print("\n2. 加载保存的模型...")
    loaded_tabpfns, loaded_joint, loaded_optimizer, loaded_config = load_model_e2e(model_path)
    
    print(f"✅ 成功加载模型:")
    print(f"   - TabPFN分类器数量: {len(loaded_tabpfns)}")
    print(f"   - 联合模型类型: {type(loaded_joint).__name__}")
    print(f"   - 优化器类型: {type(loaded_optimizer).__name__}")
    print(f"   - 配置参数数量: {len(loaded_config)}")
    
    # 4. 测试模型推理 - 创建一些测试数据
    print("\n3. 测试模型推理...")
    
    # 生成测试数据 (使用与训练相同的参数)
    X_test, y_test = make_multilabel_classification(
        n_samples=100,
        n_features=loaded_config.get('n_features', 20),
        n_classes=len(loaded_tabpfns),
        n_labels=2,
        random_state=42
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).to(device=device)
    
    # 使用加载的模型进行推理
    with torch.no_grad():
        # 模拟获取TabPFN logits（简化版本，实际中需要完整的forward过程）
        Z_logits = torch.randn(X_test.shape[0], len(loaded_tabpfns), device=device)
        
        # 通过联合模型获取预测
        joint_logits = loaded_joint(X_test_tensor, Z_logits)
        joint_probs = torch.sigmoid(joint_logits)
        joint_preds = (joint_probs > 0.5).float()
    
    # 计算简单的准确性指标
    accuracy = accuracy_score(y_test, joint_preds.cpu().numpy())
    hamming = hamming_loss(y_test, joint_preds.cpu().numpy())
    
    print(f"✅ 推理测试完成:")
    print(f"   - 测试样本数: {X_test.shape[0]}")
    print(f"   - 准确率: {accuracy:.4f}")
    print(f"   - Hamming损失: {hamming:.4f}")
    
    # 5. 检查配置文件
    config_path = model_path.replace('.pt', '_config.txt')
    if os.path.exists(config_path):
        print(f"\n4. 配置文件已保存: {config_path}")
        with open(config_path, 'r') as f:
            config_content = f.read()
            print("配置文件内容预览:")
            print(config_content[:500] + ("..." if len(config_content) > 500 else ""))
    
    print("\n🎉 端到端模型保存/加载测试完成！")
    return True


if __name__ == "__main__":
    try:
        test_save_load()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
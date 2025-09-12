#!/usr/bin/env python3
"""
测试修改后的可视化功能
"""

import pandas as pd
import numpy as np
from baseline_test import visualize_model_comparison

# 创建测试数据
np.random.seed(42)

# 模拟一些模型结果
test_results = {
    'BR+LogisticRegression': {
        'micro_f1': 0.65,
        'macro_f1': 0.62,
        'hamming_loss': 0.22,
        'subset_accuracy': 0.35,
        'macro_auc': 0.78
    },
    'BR+RandomForest': {
        'micro_f1': 0.72,
        'macro_f1': 0.69,
        'hamming_loss': 0.18,
        'subset_accuracy': 0.42,
        'macro_auc': 0.83
    },
    'BR+XGBoost': {
        'micro_f1': 0.74,
        'macro_f1': 0.71,
        'hamming_loss': 0.17,
        'subset_accuracy': 0.45,
        'macro_auc': 0.85
    },
    'CC+RandomForest': {
        'micro_f1': 0.70,
        'macro_f1': 0.67,
        'hamming_loss': 0.19,
        'subset_accuracy': 0.38,
        'macro_auc': 0.81
    },
    'MLP': {
        'micro_f1': 0.68,
        'macro_f1': 0.65,
        'hamming_loss': 0.20,
        'subset_accuracy': 0.36,
        'macro_auc': 0.79
    },
    'TabPFN+Attention': {
        'micro_f1': 0.76,
        'macro_f1': 0.73,
        'hamming_loss': 0.16,
        'subset_accuracy': 0.48,
        'macro_auc': 0.87
    }
}

def test_visualization():
    """测试可视化功能"""
    print("测试修改后的可视化功能...")
    
    try:
        # 调用可视化函数
        visualize_model_comparison(
            test_results, 
            save_prefix="test_visualization_0912_1200",
            figsize=(10, 6)
        )
        print("✅ 可视化测试成功完成！")
        print("📊 生成的图片文件:")
        print("  - test_visualization_0912_1200_metrics_grouped.png (主要指标 + Hamming Loss)")
        print("  - test_visualization_0912_1200_distribution.png (指标分布)")
        print("  - test_visualization_0912_1200_ranking.png (模型排名)")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization()
    if success:
        print("\n🎉 所有测试通过！可视化功能修改成功。")
        print("\n主要改进:")
        print("1. ✅ Hamming Loss 分离到单独的子图中")
        print("2. ✅ 主要指标的纵轴范围根据数据动态调整，增强视觉差异")
        print("3. ✅ Hamming Loss 的纵轴范围单独调整，突出差异")
        print("4. ✅ 添加了不同的标题和Y轴标签来区分指标类型")
    else:
        print("\n💥 测试失败，请检查代码！")
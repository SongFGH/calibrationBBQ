import numpy as np
from scipy.stats import rankdata


def get_auc(actual, predicted):
    """
    计算AUC值
    参数:
        - actual: 实际标签
        - predicted: 预测概率
    返回:
        - auc: AUC值
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # 检查输入是否适合AUC分析
    unique_vals = np.unique(actual)
    if len(unique_vals) != 2 or np.max(unique_vals) != 1:
        raise ValueError('AUC分析的输入异常')

    n_target = np.sum(actual == 1)
    n_background = np.sum(actual != 1)

    # 对预测值进行排序
    ranks = rankdata(predicted, method='average')  # 使用平均排名处理并列值

    # 计算AUC
    auc = (np.sum(ranks[actual == 1]) - (n_target**2 + n_target) / 2) / (n_target * n_background)

    auc = max(auc, 1 - auc)
    
    return auc

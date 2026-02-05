import numpy as np


def elbow(SV, alpha):
    """
    寻找曲线的肘部点，用于确定最优模型复杂度
    参数:
        - SV: 似然值向量
        - alpha: 阈值参数，用于确定肘部位置
    返回:
        - idxs: 符合条件的索引列表
    """
    # 对SV进行排序，保留原始索引
    sorted_indices = np.argsort(-SV)  # 降序排列
    sorted_sv = SV[sorted_indices]
    
    # 计算相邻点之间的距离变化
    diffs = np.diff(sorted_sv)
    
    # 找到变化最大的点作为肘部
    threshold = alpha * np.max(diffs)
    elbow_point = np.where(diffs > threshold)[0]
    
    if len(elbow_point) > 0:
        # 取第一个显著变化点之后的所有点
        idxs = sorted_indices[:elbow_point[0]+2]  # +2 是为了确保包含肘部点
    else:
        # 如果没有明显肘部，返回所有点的索引
        idxs = sorted_indices

    # 确保返回的是按升序排列的索引
    idxs = np.sort(idxs)
    
    # 转换为Python索引（从0开始），MATLAB是从1开始
    # 但这里我们直接返回索引，因为Python中的使用方式会自然地处理索引
    return idxs

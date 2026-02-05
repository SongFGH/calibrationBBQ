import numpy as np


def get_rmse(y, p):
    """
    计算均方根误差
    参数:
        - y: 真实值
        - p: 预测值
    返回:
        - res: RMSE值
    """
    y = np.array(y).flatten()
    p = np.array(p).flatten()
    
    res = np.sqrt(np.dot((y - p).T, (y - p)) / len(y))
    
    return res

import numpy as np
from getRMSE import get_rmse
from getAUC import get_auc
from getMCE import get_mce
from getECE import get_ece


def get_measures(PTE, YTE):
    """
    计算多种评估指标
    参数:
        - PTE: 预测概率
        - YTE: 真实标签
    返回:
        - res: 包含各种指标的字典
    """
    PTE = np.array(PTE).flatten()
    YTE = np.array(YTE).flatten()
    
    # 检查是否有NaN值
    if np.sum(np.isnan(PTE)) > 0:
        print('预测值中有一些NaN值')
    
    if np.sum(np.isnan(YTE)) > 0:
        print('标签中有一些NaN值')
    
    # 移除包含NaN值的样本
    idx = np.logical_or(np.isnan(YTE), np.isnan(PTE))
    YTE_clean = YTE[~idx]
    PTE_clean = PTE[~idx]
    
    # 计算各项指标
    res = {}
    res['RMSE'] = get_rmse(YTE_clean, PTE_clean)
    res['AUC'] = get_auc(YTE_clean, PTE_clean)
    res['ACC'] = 1 - np.sum(YTE_clean != (PTE_clean >= 0.5)) / len(YTE_clean)
    res['MCE'] = get_mce(YTE_clean, PTE_clean)
    res['ECE'] = get_ece(YTE_clean, PTE_clean)
    
    return res

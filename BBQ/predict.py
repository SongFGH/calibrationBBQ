import numpy as np


def predict(bbq, PTE, option):
    """
    该函数用于校准概率
    参数:
        - bbq: BBQ模型
        - PTE: 未校准概率向量
        - option: 0 使用模型选择, 1 使用模型平均
    返回:
        - out: 校准后的概率向量
    """
    PTE = np.array(PTE)
    out = np.zeros((len(PTE), 1))
    BBQModel = bbq.prunedModel  # 也可以使用非剪枝版本 bbq.model

    if option == 1:  # 使用模型平均
        for i in range(len(PTE)):
            out[i, :] = getMA_local(BBQModel, PTE[i])
    elif option == 0:  # 使用模型选择
        out = getHistPr_local(BBQModel[0].histModel, BBQModel[0].cutPoints, PTE)

    return out


def getMA_local(BBQModel, x):
    """
    执行模型平均，通过组合多个BBQ模型的预测结果来获得校准概率
    参数:
        - BBQModel: 包含多个BBQ模型的列表
        - x: 需要校准的输入概率值
    返回:
        - res: 使用模型平均方法得到的校准概率
    """
    N = len(BBQModel)
    p = np.zeros((N, 1))
    SV = BBQModel[0].SV  # 已经是相对似然度

    for i in range(N):
        p[i, :] = getHistPr_local(BBQModel[i].histModel, BBQModel[i].cutPoints, x)

    SV = np.array(SV)  # 将SV转换为numpy数组
    res = np.dot(SV.T, p) / np.sum(SV)
    return res


def getHistPr_local(histModel, cutPoints, PTE):
    """
    执行基于局部直方图的概率校准，通过找到每个输入概率的适当区间并返回相应的校准概率
    参数:
        - histModel: 直方图模型，包含不同区间的概率映射
        - cutPoints: 定义直方图区间边界的分割点
        - PTE: 需要处理的未校准概率向量
    返回:
        - res: 应用直方图映射后得到的校准概率向量
    """
    # 检查PTE是否为标量值，如果是则转换为数组
    if np.isscalar(PTE):
        N = 1
        PTE = np.array([PTE])
    else:
        N = len(PTE)
    
    B = len(histModel)
    cutPoints = np.concatenate(([0], cutPoints, [1]))
    res = np.zeros((N, 1))

    for i in range(N):
        x = PTE[i]
        minIdx = 0  # Python索引从0开始
        maxIdx = B

        # 二分查找确定x属于哪个区间
        while (maxIdx - minIdx) > 1:
            midIdx = int(np.floor((minIdx + maxIdx) / 2))
            if x > cutPoints[midIdx]:
                minIdx = midIdx
            elif x < cutPoints[midIdx]:
                maxIdx = midIdx
            else:
                minIdx = midIdx
                break

        idx = minIdx
        res[i, :] = histModel[idx].P

        # 处理特殊情况：当有多个区间具有完全相同的最小-最大范围但具有不同概率值时
        # (这在朴素贝叶斯中极少数情况下发生)
        cnt = 1
        k = idx - 1
        while k >= 0:
            if (hasattr(histModel[k], 'min') and hasattr(histModel[idx], 'min') and 
                histModel[k].min == histModel[idx].min and 
                hasattr(histModel[k], 'max') and hasattr(histModel[idx], 'max') and 
                histModel[k].max == histModel[idx].max):
                
                res[i, :] = res[i, :] + histModel[k].P
                k = k - 1
                cnt = cnt + 1
            else:
                break

        k = idx + 1
        while k < B:
            if (hasattr(histModel[k], 'min') and hasattr(histModel[idx], 'min') and 
                histModel[k].min == histModel[idx].min and 
                hasattr(histModel[k], 'max') and hasattr(histModel[idx], 'max') and 
                histModel[k].max == histModel[idx].max):
                
                res[i, :] = res[i, :] + histModel[k].P
                k = k + 1
                cnt = cnt + 1
            else:
                break

        res[i, :] = res[i, :] / cnt

    # 如果原始输入是标量，则返回标量值
    if N == 1 and np.isscalar(x):
        return res[0, :]
    return res

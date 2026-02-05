import numpy as np


def get_ece(y, p):
    """
    计算Expected Calibration Error
    参数:
        - y: 真实标签
        - p: 预测概率
    返回:
        - res: ECE值
    """
    # 将输入的预测概率p转换为numpy数组并展平为一维
    predictions = np.array(p).flatten()
    # 将输入的真实标签y转换为numpy数组并展平为一维
    labels = np.array(y).flatten()
    
    # 将预测值和标签合并成二维数组，每行包含一个预测值和对应的真实标签
    ordered = np.column_stack((predictions, labels))
    # 按照预测值（第一列）对合并后的数组进行升序排序
    ordered = ordered[np.argsort(ordered[:, 0])]
    
    # 获取总样本数量
    N = ordered.shape[0]
    # 计算总样本数除以10的余数，用于处理不能整除的情况
    rest = N % 10
    # 确定桶的数量，取样本数和10的最小值
    B = min(N, 10)
    
    # 创建一个长度为B的零数组，用于存储每个桶的校准误差
    s = np.zeros(B)
    
    # 初始化权重数组
    W = np.zeros((B, 1))
    
    # 循环遍历每个桶
    for i in range(B):
        # 对于前rest个桶，使用较大的大小以处理不能整除的情况
        if i < rest:
            # 计算当前桶的起始索引
            start_idx = i * int(np.ceil(N / 10))
            # 计算当前桶的结束索引
            end_idx = (i + 1) * int(np.ceil(N / 10))
            # 提取当前桶的数据
            group = ordered[start_idx:end_idx, :]
        else:
            # 对于其余的桶，使用较小的大小
            # 计算当前桶的起始索引（考虑前面较大桶的影响）
            start_idx = rest + i * int(np.floor(N / 10))
            # 计算当前桶的结束索引
            end_idx = rest + (i + 1) * int(np.floor(N / 10))
            # 提取当前桶的数据
            group = ordered[start_idx:end_idx, :]
        
        # 获取当前桶的样本数量
        n = group.shape[0]
        # 如果当前桶不为空
        if n > 0:
            # 计算当前桶中实际正类的比例（真实标签的平均值）
            observed = np.mean(group[:, 1])
            # 计算当前桶中预测概率的平均值
            expected = np.mean(group[:, 0])
            # 计算当前桶的平均校准误差（预测概率与实际准确率的差的绝对值）
            s[i] = abs(expected - observed)
            # 计算当前桶的权重（当前桶样本数占总样本数的比例）
            W[i] = n / N
        else:
            # 如果桶为空，则误差设为0，权重也为0
            s[i] = 0
            W[i] = 0
    
    # 计算加权平均校准误差，即期望校准误差(ECE)
    res = np.dot(s.T, W.flatten())
    
    return res


# 测试代码
if __name__ == "__main__":
    # 示例1：完全校准的模型（预测概率等于实际准确率）
    print("示例1：完全校准的模型")
    perfect_predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    perfect_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 假设这些标签接近预测概率
    ece_perfect = get_ece(perfect_labels, perfect_predictions)
    print(f"ECE (完美校准): {ece_perfect:.4f}")
    
    # 示例2：完全不校准的模型（预测概率远偏离实际准确率）
    print("\n示例2：不校准的模型")
    bad_predictions = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9]
    bad_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 预测与实际相反
    ece_bad = get_ece(bad_labels, bad_predictions)
    print(f"ECE (不校准): {ece_bad:.4f}")
    
    # 示例3：随机生成数据
    print("\n示例3：随机生成的数据")
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    random_predictions = np.random.rand(100)  # 生成100个0到1之间的随机预测概率
    random_labels = np.random.randint(0, 2, size=100)  # 生成100个随机标签(0或1)
    ece_random = get_ece(random_labels, random_predictions)
    print(f"ECE (随机数据): {ece_random:.4f}")
    
    # 示例4：极端情况 - 预测全部正确
    print("\n示例4：极端情况 - 高准确率预测")
    extreme_predictions = [0.01, 0.02, 0.03, 0.97, 0.98, 0.99]
    extreme_labels = [0, 0, 0, 1, 1, 1]  # 预测基本正确
    ece_extreme = get_ece(extreme_labels, extreme_predictions)
    print(f"ECE (极端情况): {ece_extreme:.4f}")
    
    # 示例5：小数据集测试
    print("\n示例5：小数据集测试")
    small_predictions = [0.2, 0.8]
    small_labels = [0, 1]
    ece_small = get_ece(small_labels, small_predictions)
    print(f"ECE (小数据集): {ece_small:.4f}")

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


class ScaleData:
    """
    存储缩放数据的类
    """
    def __init__(self, scale_factor=None, shift=None):
        self.scale_factor = scale_factor
        self.shift = shift


def kernel_function(X, Y, kernel_type='rbf', **kwargs):
    """
    计算核函数
    参数:
        - X: 第一个数据矩阵
        - Y: 第二个数据矩阵
        - kernel_type: 核函数类型 ('rbf', 'poly', 'linear')
        - **kwargs: 核函数参数
    返回:
        - K: 核矩阵
    """
    if kernel_type == 'rbf':
        gamma = kwargs.get('gamma', 'scale')
        return rbf_kernel(X, Y, gamma=gamma)
    elif kernel_type == 'poly':
        degree = kwargs.get('degree', 3)
        coef0 = kwargs.get('coef0', 1)
        gamma = kwargs.get('gamma', 1)
        return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel_type == 'linear':
        return linear_kernel(X, Y)
    else:
        # 默认使用RBF核
        gamma = kwargs.get('gamma', 'scale')
        return rbf_kernel(X, Y, gamma=gamma)


def Mysvmdecision(Xnew, svm_struct):
    """
    计算SVM决策函数
    参数:
        - Xnew: 新的输入数据
        - svm_struct: SVM结构体
    返回:
        - out: 决策函数输出
        - f: 决策函数值
    """
    # 提取SVM结构中的参数
    sv = svm_struct.SupportVectors
    alpha_hat = svm_struct.Alpha
    bias = svm_struct.Bias
    
    # 确定核函数类型和参数
    kernel_type = getattr(svm_struct, 'KernelType', 'rbf')
    kernel_args = getattr(svm_struct, 'KernelFunctionArgs', {})
    
    # 计算核函数值
    k_matrix = kernel_function(sv, Xnew, kernel_type=kernel_type, **kernel_args)
    
    # 计算决策函数值
    f = np.dot(k_matrix.T, alpha_hat) + bias
    
    # 返回决策函数值（与MATLAB版本一致）
    out = f
    
    return out, f

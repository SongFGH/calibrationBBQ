import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.io import loadmat  # 用于加载.mat文件
import os

# 导入自定义模块
from build import build
from predict import predict
from getMeasures import get_measures


def load_data():
    """
    加载数据集 (模拟 load data; MATLAB命令)
    """
    # 尝试加载.mat格式的数据文件
    mat_file_path = '/Users/11179767/Code/test/py/BBQ/data.mat'  # 相对于当前脚本位置的路径
    if os.path.exists(mat_file_path):
        data = loadmat(mat_file_path)
        # 假设.mat文件中包含XTR, YTR, XTE, YTE变量
        XTR, YTR = data['XTR'], data['YTR'].ravel()  # 使用ravel()展平标签数组
        XTE, YTE = data['XTE'], data['YTE'].ravel()
    else:
        # 如果没有找到.mat文件，检查是否有其他格式的数据文件
        print(f"警告: 未找到 {mat_file_path} 文件，使用示例数据")
        # 使用示例数据（需要替换为真实数据）
        X, y = make_classification(n_samples=1000, n_features=20, n_redundant=10,
                                   n_informative=10, random_state=42, n_clusters_per_class=1)
        XTR, XTE, YTR, YTE = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return XTR, YTR, XTE, YTE


def sigmoid(x):
    """Sigmoid函数，将SVM输出转换为概率"""
    return np.exp(x) / (1 + np.exp(x))


def test_bbq_with_svm():
    """
    测试BBQ模型与SVM结合的校准效果 (与MATLAB版本test.m逻辑一致)
    """
    # 加载数据集
    print("加载数据...")
    XTR, YTR, XTE, YTE = load_data()
    
    # %% 应用线性SVM回归模型进行预测
    print('Using Linear SVM ')
    
    # 训练线性SVM模型
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(XTR, YTR)
    
    # 使用训练集进行预测并转换为未经校准的概率
    PTR = svm_model.decision_function(XTR)  # 获取决策函数值
    PTR = sigmoid(PTR)  # 将SVM输出转换为未经校准的概率值（sigmoid变换）
    
    # 使用测试集进行预测并转换为未经校准的概率
    PTE = svm_model.decision_function(XTE)  # 获取决策函数值
    PTE = sigmoid(PTE)  # 将SVM输出转换为未经校准的概率值（sigmoid变换）
    
    print('Performance of (Linear) SVM probabilities : ')
    M = get_measures(PTE, YTE)  # 获取未经校准的SVM概率的性能度量
    print(M)
    
    # 构建BBQ模型
    options = {'N0': 2}  # 设置BBQ模型的先验参数
    BBQ = build(PTR, YTR, options)  # 使用训练集预测和真实标签构建BBQ校准器
    PTE_bbq = predict(BBQ, PTE, 1)  # 使用BBQ校准测试集预测概率
    print('Performance of Calibrated Probabilities using BBQ : ')
    M_bbq = get_measures(PTE_bbq, YTE)  # 获取经BBQ校准的概率的性能度量
    print(M_bbq)
    
    # %% 应用二次SVM回归模型进行预测
    print('Using Quadratic SVM ')
    
    # 训练二次核SVM模型
    quad_svm_model = SVC(kernel='poly', degree=2, probability=True, random_state=42)
    quad_svm_model.fit(XTR, YTR)
    
    # 使用训练集进行预测并转换为未经校准的概率
    PTR_quad = quad_svm_model.decision_function(XTR)  # 获取决策函数值
    PTR_quad = sigmoid(PTR_quad)  # 将SVM输出转换为未经校准的概率值（sigmoid变换）
    
    # 使用测试集进行预测并转换为未经校准的概率
    PTE_quad = quad_svm_model.decision_function(XTE)  # 获取决策函数值
    PTE_quad = sigmoid(PTE_quad)  # 将SVM输出转换为未经校准的概率值（sigmoid变换）
    
    print('Performance of (Quadratic) SVM probabilities : ')
    M_quad = get_measures(PTE_quad, YTE)  # 获取未经校准的二次SVM概率的性能度量
    print(M_quad)
    
    # 构建BBQ模型
    options = {'N0': 2}  # 设置BBQ模型的先验参数
    BBQ_quad = build(PTR_quad, YTR, options)  # 使用训练集预测和真实标签构建BBQ校准器
    
    # 使用BBQ校准预测
    PTE_bbq_quad = predict(BBQ_quad, PTE_quad, 1)  # 使用BBQ校准测试集预测概率
    print('Performance of Calibrated Probabilities using BBQ : ')
    M_bbq_quad = get_measures(PTE_bbq_quad, YTE)  # 获取经BBQ校准的概率的性能度量
    print(M_bbq_quad)
    
    print('End!')


if __name__ == "__main__":
    test_bbq_with_svm()

import numpy as np
from sklearn.svm import SVC
from .Mysvmdecision import Mysvmdecision


class SVMStruct:
    """
    用于存储SVM模型信息的类，模拟MATLAB中的svmStruct结构
    """
    def __init__(self, model, group_names, scale_data=None, figure_handles=None):
        self.model = model
        self.GroupNames = group_names
        self.ScaleData = scale_data
        self.FigureHandles = figure_handles


def Mysvmclassify(svm_struct, sample, show_plot=False):
    """
    使用支持向量机分类器对数据进行分类
    参数:
        - svm_struct: SVM模型结构
        - sample: 待分类的样本数据
        - show_plot: 是否显示绘图
    返回:
        - outclass: 分类结果
    """
    # 检查输入是否为有效的svm_struct
    if not isinstance(svm_struct, SVMStruct):
        raise ValueError("svm_struct应该是由Mysvmtrain创建的结构")

    # 将样本转换为numpy数组
    sample = np.asarray(sample)
    
    # 如果需要，对数据进行缩放
    if svm_struct.ScaleData is not None:
        scale_factor = svm_struct.ScaleData.get('scale_factor', 1)
        shift = svm_struct.ScaleData.get('shift', 0)
        sample = scale_factor * (sample + shift)

    # 进行分类决策
    try:
        # 使用Mysvmdecision函数（这是修改过的版本，符合项目需求）
        classified = Mysvmdecision(sample, svm_struct)
    except Exception as e:
        raise RuntimeError(f"分类过程中遇到错误: {str(e)}")

    # 如果有未分类的数据点，处理它们
    outclass = classified.copy()
    unclassified = np.isnan(outclass)
    
    if np.any(unclassified):
        print("警告: 一些样本无法分类。这可能是由于数据中的NaN值导致的。")
        # 对于未分类的点，将其标记为0
        outclass[unclassified] = 0

    # 根据代码最后的处理，返回负值
    outclass = -outclass
    
    return outclass

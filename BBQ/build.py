import numpy as np
from scipy.special import gammaln
from elbow import elbow  # 修改为绝对导入


class BBQModel:
    def __init__(self):
        self.model = None
        self.prunedModel = None


class ModelItem:
    def __init__(self):
        self.histModel = None
        self.cutIdx = None
        self.cutPoints = None
        self.score = None
        self.logLikelihood = None
        self.SV = None
        self.maxScoreIdx = None
        self.minScoreIdx = None


class HistModelItem:
    def __init__(self):
        self.min = None
        self.max = None
        self.n = None
        self.n1 = None
        self.n0 = None
        self.P = None


def build(PTR, YTR, options=None):
    """
    构建BBQ模型
    参数:
        - PTR: 预测值向量
        - YTR: 真实标签向量 {0,1}
        - options: 可选参数结构体
    返回:
        - bbq: BBQ模型对象
    """
    if options is None:
        options = {}

    # 获取选项参数
    scoring_func = options.get('scoringFunc', 'BDeu2')
    N0 = options.get('N0', 2)
    alpha = options.get('alpha', 0.001)
    run_sort = options.get('sort', 1)

    PTR = np.array(PTR)
    YTR = np.array(YTR)

    if run_sort == 1:
        idx = np.argsort(PTR)
        PTR = PTR[idx]
        YTR = YTR[idx]

    N = len(PTR)
    lnfact = initlnfact_local(N + 1)  # 包含log(n!)的数组，从0到N

    max_bin_no = min(int(np.ceil(N / 5)), int(np.ceil(10 * N ** (1/3))))  # 至少每个bin有5个实例
    min_bin_no = max(1, int(np.floor(N ** (1/3) / 10)))  # 至少1个bin
    mnm = max_bin_no - min_bin_no + 1  # 最大模型数
    model = [None] * mnm

    # 根据评分函数设置回调函数
    if scoring_func == 'BDeu':
        score_func = get_bdeu_score_local
    elif scoring_func == 'BDeu2':  # 论文中使用的N0 = 2B版本
        score_func = get_bdeu_score2_local

    opt1 = {
        'PTR': PTR,
        'lnfact': lnfact,
        'N0': N0
    }

    # 串行处理（替代MATLAB的parfor）
    for b in range(mnm):
        hist_model, cut_idx, cut_points, log_likelihood = hist_calibration_freq_local(
            PTR, YTR, b + min_bin_no)
        func_opt = build_func_opt_local(opt1, hist_model, cut_idx, cut_points, log_likelihood)
        score = score_func(func_opt)

        model[b] = ModelItem()
        model[b].histModel = hist_model
        model[b].cutIdx = cut_idx
        model[b].cutPoints = cut_points
        model[b].score = score
        model[b].logLikelihood = log_likelihood

    # 计算相对似然度
    max_score = -np.inf
    max_score_idx = 0
    min_score = np.inf
    min_score_idx = 0
    sv = np.zeros(mnm)
    for b in range(mnm):
        sv[b] = model[b].score
        if model[b].score > max_score:
            max_score_idx = b
            max_score = model[b].score

        if model[b].score < min_score:
            min_score_idx = b
            min_score = model[b].score

    sv = np.exp((np.min(sv) - sv) / 2)  # 计算相对似然度

    model[0].maxScoreIdx = max_score_idx  # 用于模型选择
    model[0].minScoreIdx = min_score_idx  # 用于模型选择
    model[0].SV = sv

    idxs = elbow(sv, alpha)
    model2 = process_model_local(model, idxs)

    bbq = BBQModel()
    bbq.model = model
    bbq.prunedModel = model2

    return bbq


def process_model_local(in_model, idxs):
    """处理模型，提取指定索引的模型"""
    out_model = []
    for i in range(len(idxs)):
        out_model.append(in_model[idxs[i]])
    out_model[0].minScoreIdx = 1
    out_model[0].SV = [in_model[0].SV[i] for i in idxs]
    return out_model


def build_func_opt_local(opt, hist_model, cut_idx, cut_points, log_likelihood):
    """构建函数选项"""
    n = len(opt['PTR'])
    k = len(hist_model)

    func_opt = {
        'histModel': hist_model,
        'cutIdx': cut_idx,
        'cutPoints': cut_points,
        'logLikelihood': log_likelihood,
        'K': k,
        'N': n,
        'PTR': opt['PTR'],
        'lnfact': opt['lnfact'],
        'N0': opt['N0']
    }
    return func_opt


def get_bdeu_score_local(opt):
    """BDeu评分函数"""
    hist_model = opt['histModel']

    # 首先计算对数边际似然
    b = len(hist_model)
    n0 = opt['N0']
    c = n0 / b

    score = b * gammaln(c)
    for j in range(b):
        nj = hist_model[j].n
        nj0 = hist_model[j].n0
        nj1 = hist_model[j].n1
        pj = 0.5
        score = score + gammaln(nj1 + c * pj) + gammaln(nj0 + c * (1 - pj)) - gammaln(nj + c) \
                      - gammaln(c * pj) - gammaln(c * (1 - pj))

    score = -2 * score
    return score


def get_bdeu_score2_local(opt):
    """BDeu2评分函数（论文中使用的版本）"""
    hist_model = opt['histModel']

    # 首先计算对数边际似然
    b = len(hist_model)
    n0 = 2 * b  # N0 = 2B
    c = n0 / b

    score = b * gammaln(c)
    for j in range(b):
        nj = hist_model[j].n
        nj0 = hist_model[j].n0
        nj1 = hist_model[j].n1
        pj = (hist_model[j].min + hist_model[j].max) / 2
        pj = min(pj, 1 - 5e-3)
        pj = max(pj, 5e-3)
        score = score + gammaln(nj1 + c * pj) + gammaln(nj0 + c * (1 - pj)) - gammaln(nj + c) \
                      - gammaln(c * pj) - gammaln(c * (1 - pj))

    score = -2 * score
    return score


def initlnfact_local(n):
    """初始化对数阶乘数组"""
    lnfact = np.zeros(n + 1)
    for w in range(2, n + 1):
        lnfact[w] = lnfact[w - 1] + np.log(w - 1)
    return lnfact


def hist_calibration_freq_local(PTR, YTR, b):
    """直方图校准频率函数"""
    N = len(YTR)
    log_likelihood = 0

    if b == 1:
        hist_model = [HistModelItem()]
        hist_model[0].min = 0
        hist_model[0].max = 1

        # 构建Beta平滑的直观先验
        m0 = (hist_model[0].min + hist_model[0].max) / 2
        idx = (YTR == 1)
        ptr1 = PTR[idx]
        p0 = (np.sum(ptr1) + m0) / (len(ptr1) + 1)

        hist_model[0].n = len(YTR)
        hist_model[0].n1 = np.sum(YTR)
        hist_model[0].n0 = hist_model[0].n - hist_model[0].n1
        hist_model[0].P = (hist_model[0].n1 + p0) / (hist_model[0].n + 1)

        if hist_model[0].n1 > 0:
            log_likelihood += hist_model[0].n1 * np.log(hist_model[0].P)
        if hist_model[0].n0 > 0:
            log_likelihood += hist_model[0].n0 * np.log(1 - hist_model[0].P)

        cut_idx = []
        cut_points = []

        return hist_model, cut_idx, cut_points, log_likelihood
    else:  # 当b > 1时
        # 初始化cut_idx数组，避免UnboundLocalError
        cut_idx = np.array([0])  # 初始化cut_idx数组
        
        max_nj = 0
        yhat = PTR
        y = YTR
        c = int(np.floor(len(y) / b))
        i = 1
        idx = 1

        t_list = []
        while i < b:
            idx1 = (i - 1) * c + 1
            idx2 = i * c
            j = i + 1

            while j <= b:
                if j < b:
                    jidx2 = j * c
                    if PTR[jidx2 - 1] == PTR[idx1 - 1]:  # MATLAB索引从1开始，Python从0开始
                        idx2 = jidx2
                        j = j + 1
                    else:
                        break
                else:
                    jidx2 = N
                    if PTR[jidx2 - 1] == PTR[idx1 - 1]:
                        idx2 = jidx2
                        j = j + 1
                    else:
                        break
                j = j + 1

            t_item = {
                'Y': y[idx1 - 1:idx2],
                'PTR': PTR[idx1 - 1:idx2],
                'Yhat': yhat[idx1 - 1:idx2]
            }
            t_list.append(t_item)
            max_nj = max(max_nj, idx2 - idx1 + 1)

            if idx2 < N:
                if idx > len(cut_idx):  # Python中动态扩展数组
                    cut_idx = np.append(cut_idx, [idx2])
                else:
                    cut_idx[idx - 1] = idx2
            idx = idx + 1
            i = j

        if idx2 < N:
            t_item = {
                'Y': y[idx2:],  # MATLAB中idx2+1:end，Python中idx2:（注意偏移）
                'PTR': PTR[idx2:],
                'Yhat': yhat[idx2:]
            }
            t_list.append(t_item)

        b0 = b
        b = len(t_list)
        hist_model = [HistModelItem() for _ in range(len(t_list))]

        hist_model[0].min = 0
        hist_model[0].max = (t_list[0]['Yhat'][-1] + t_list[1]['Yhat'][0]) / 2
        cut_points = [hist_model[0].max]

        # 构建Beta平滑的直观先验
        m0 = (hist_model[0].min + hist_model[0].max) / 2
        idx = (t_list[0]['Y'] == 1)
        ptr1 = t_list[0]['PTR'][idx]
        p0 = (np.sum(ptr1) + m0) / (len(ptr1) + 1)

        hist_model[0].n = len(t_list[0]['Y'])
        hist_model[0].n1 = np.sum(t_list[0]['Y'])
        hist_model[0].n0 = hist_model[0].n - hist_model[0].n1
        hist_model[0].P = (hist_model[0].n1 + p0) / (hist_model[0].n + 1)

        if hist_model[0].n1 > 0:
            log_likelihood += hist_model[0].n1 * np.log(hist_model[0].P)
        if hist_model[0].n0 > 0:
            log_likelihood += hist_model[0].n0 * np.log(1 - hist_model[0].P)

        for i in range(1, b - 1):
            hist_model[i].min = (t_list[i]['Yhat'][0] + t_list[i - 1]['Yhat'][-1]) / 2
            hist_model[i].max = (t_list[i]['Yhat'][-1] + t_list[i + 1]['Yhat'][0]) / 2
            cut_points.append(hist_model[i].max)

            # 构建Beta平滑的直观先验
            m0 = (hist_model[i].min + hist_model[i].max) / 2
            idx = (t_list[i]['Y'] == 1)
            ptr1 = t_list[i]['PTR'][idx]
            p0 = (np.sum(ptr1) + m0) / (len(ptr1) + 1)

            hist_model[i].n = len(t_list[i]['Y'])
            hist_model[i].n1 = np.sum(t_list[i]['Y'])
            hist_model[i].n0 = hist_model[i].n - hist_model[i].n1
            hist_model[i].P = (hist_model[i].n1 + p0) / (hist_model[i].n + 1)

            if hist_model[i].n1 > 0:
                log_likelihood += hist_model[i].n1 * np.log(hist_model[i].P)
            if hist_model[i].n0 > 0:
                log_likelihood += hist_model[i].n0 * np.log(1 - hist_model[i].P)

        hist_model[b - 1].min = (t_list[b - 1]['Yhat'][0] + t_list[b - 2]['Yhat'][-1]) / 2
        hist_model[b - 1].max = 1

        # 构建Beta平滑的直观先验
        m0 = (hist_model[b - 1].min + hist_model[b - 1].max) / 2
        idx = (t_list[b - 1]['Y'] == 1)
        ptr1 = t_list[b - 1]['PTR'][idx]
        p0 = (np.sum(ptr1) + m0) / (len(ptr1) + 1)

        hist_model[b - 1].n = len(t_list[b - 1]['Y'])
        hist_model[b - 1].n1 = np.sum(t_list[b - 1]['Y'])
        hist_model[b - 1].n0 = hist_model[b - 1].n - hist_model[b - 1].n1
        hist_model[b - 1].P = (hist_model[b - 1].n1 + p0) / (hist_model[b - 1].n + 1)

        if hist_model[b - 1].n1 > 0:
            log_likelihood += hist_model[b - 1].n1 * np.log(hist_model[b - 1].P)
        if hist_model[b - 1].n0 > 0:
            log_likelihood += hist_model[b - 1].n0 * np.log(1 - hist_model[b - 1].P)

        # 处理cut_idx
        cut_idx = [int(ci) for ci in cut_idx] if len(cut_idx) > 0 else []

        return hist_model, cut_idx, cut_points, log_likelihood

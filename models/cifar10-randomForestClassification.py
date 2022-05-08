from re import X
from time import time
import numpy as np
from numpy import *

from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn import decomposition
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from tqdm import *


def load_cifar(train_batch=5):
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []

    def unpickle(file):
        with open(file, "rb") as fo:
            data = pickle.load(fo, encoding="latin1")
        return data

    for i in range(train_batch):
        batchName = "cifar10/train/cifar-10-batches-py/data_batch_{0}".format(i + 1)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled["data"])
        trn_labels.extend(unpickled["labels"])
    unpickled = unpickle("cifar10/train/cifar-10-batches-py/test_batch")
    tst_data.extend(unpickled["data"])
    tst_labels.extend(unpickled["labels"])
    trn_data = np.array(trn_data)
    trn_labels = np.array(trn_labels)
    tst_data = np.array(tst_data)
    tst_labels = np.array(tst_labels)
    return (
        (trn_data - trn_data.mean(axis=0)),
        trn_labels,
        (tst_data - trn_data.mean(axis=0)),
        tst_labels,
    )


def pca_func(trn_data, tst_data):
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(trn_data)
    return pca.transform(trn_data), pca.transform(tst_data)

# 切分数据集,实现交叉验证。
def splitDataSet(dataSet, n_folds):
    fold_size = len(dataSet) / n_folds
    data_split = []
    begin = 0
    end = fold_size
    for i in range(n_folds):
        data_split.append(dataSet[begin:end, :])
        begin = end
        end += fold_size
    return data_split


# 构建n个子集
def get_subsamples(dataSet, n):
    subDataSet = []
    for i in range(n):
        index = []
        for k in range(len(dataSet)):
            index.append(np.random.randint(len(dataSet)))
        subDataSet.append(dataSet[index, :])
    return subDataSet


# 划分数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] < value)[0], :]
    return mat0, mat1


# 计算方差,回归时使用
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * shape(dataSet)[0]


# 计算平均值,回归时使用
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def most_number(dataSet):  # 返回多类
    # number=set(dataSet[:,-1])
    ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tp = dict.fromkeys(ls, 0)
    for data in dataSet:
        tp[data[-1]] += 1
    rk = sorted(tp.items(), key=lambda x: x[1], reverse=True)
    return rk[0][0]


# 计算基尼指数
def gini(dataSet):
    corr = 0.0
    for i in set(dataSet[:, -1]):
        corr += (len(np.nonzero(dataSet[:, -1] == i)[0]) / len(dataSet)) ** 2
    return 1 - corr


# 选取任意的m个特征,在这m个特征中,选取分割时的最优特征
def select_best_feature(dataSet, m, alpha="regression"):
    f = dataSet.shape[1]
    index = []
    bestS = inf
    bestfeature = 0
    bestValue = 0
    if alpha == "regression":
        S = regErr(dataSet)
    else:
        S = gini(dataSet)
    for i in range(m):
        index.append(np.random.randint(f - 1))
    print(index)
    for feature in index:
        for splitVal in set(dataSet[:, feature]):
            mat0, mat1 = binSplitDataSet(dataSet, feature, splitVal)
            if alpha == "regression":
                newS = regErr(mat0) + regErr(mat1)
            else:
                newS = gini(mat0) + gini(mat1)
            if bestS > newS:
                bestfeature = feature
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < 0.001 and alpha == "regression":  # 如果误差不大就退出
        return None, regLeaf(dataSet)
    elif (S - bestS) < 0.001:
        # print(S,bestS)
        return None, most_number(dataSet)
    # mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
    return bestfeature, bestValue


def createTree(dataSet, alpha="regression", m=20, max_level=5):  # 实现决策树,使用20个特征,深度为10
    bestfeature, bestValue = select_best_feature(dataSet, m, alpha=alpha)
    if bestfeature == None:
        return bestValue

    retTree = {}
    max_level -= 1
    if max_level < 0:  # 控制深度
        return regLeaf(dataSet)
    retTree["bestFeature"] = bestfeature
    retTree["bestVal"] = bestValue
    lSet, rSet = binSplitDataSet(dataSet, bestfeature, bestValue)
    retTree["right"] = createTree(rSet, alpha, m, max_level)
    retTree["left"] = createTree(lSet, alpha, m, max_level)
    # print('retTree:',retTree)
    return retTree


def RondomForest(X, label, n, alpha="regression"):  # 树的个数
    # dataSet=get_Datasets()
    Trees = []
    for i in range(n):
        print("tree:", i)
        X_train, X_test, y_train, y_test = train_test_split(
            X, label, test_size=0.33, random_state=42
        )
        X_train = np.concatenate((X_train, y_train.reshape((-1, 1))), axis=1)
        Trees.append(createTree(X_train, alpha=alpha))
        print("done")
    return Trees


# 预测单个数据样本
def treeForecast(tree, data, alpha="regression"):
    if alpha == "regression":
        if not isinstance(tree, dict):
            return float(tree)
        if data[tree["bestFeature"]] > tree["bestVal"]:
            if type(tree["left"]) == "float":
                return tree["left"]
            else:
                return treeForecast(tree["left"], data, alpha)
        else:
            if type(tree["right"]) == "float":
                return tree["right"]
            else:
                return treeForecast(tree["right"], data, alpha)
    else:
        if not isinstance(tree, dict):
            return int(tree)
        if data[tree["bestFeature"]] > tree["bestVal"]:
            if type(tree["left"]) == "int":
                return tree["left"]
            else:
                return treeForecast(tree["left"], data, alpha)
        else:
            if type(tree["right"]) == "int":
                return tree["right"]
            else:
                return treeForecast(tree["right"], data, alpha)


# 单棵树预测测试集
def createForeCast(tree, dataSet, alpha="regression"):
    m = len(dataSet)
    yhat = np.mat(zeros((m, 1)))
    for i in range(m):
        yhat[i, 0] = treeForecast(tree, dataSet[i, :], alpha)
    return yhat.A


# 随机森林预测
def predictTree(Trees, dataSet, alpha="regression"):
    m = len(dataSet)
    if alpha == "regression":
        yhat = np.mat(zeros((m, 1)))
        for tree in Trees:
            yhat += createForeCast(tree, dataSet, alpha)
        yhat /= len(Trees)
        return yhat
    else:
        ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        n = len(Trees)
        yhat = np.zeros((n, m))
        i = 0
        for tree in Trees:
            yhat[i, :] = np.squeeze(
                createForeCast(tree, dataSet, alpha)
            )  # .reshape(-1,1)
            i += 1
        print(yhat)
        for col in range(m):
            tp = dict.fromkeys(ls, 0)
            for row in range(n):
                tp[yhat[row, col]] += 1
            rk = sorted(tp.items(), key=lambda x: x[1], reverse=True)
            yhat[0, col] = rk[0][0]
        return yhat[0]


# 计算各项评价指标
def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame(
        {"Label": labels, "Precision": p, "Recall": r, "F1": f1, "Support": s}
    )
    res2 = pd.DataFrame(
        {
            "Label": ["总体"],
            "Precision": [tot_p],
            "Recall": [tot_r],
            "F1": [tot_f1],
            "Support": [tot_s],
        }
    )
    res2.index = [999]
    res = pd.concat([res1, res2])

    # 计算混淆矩阵
    conf_mat = pd.DataFrame(
        confusion_matrix(y_true, y_pred), columns=labels, index=labels
    )
    return conf_mat, res[["Label", "Precision", "Recall", "F1", "Support"]]


if __name__ == "__main__":
    method_dim_red = "PCA"
    cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels = load_cifar(
        train_batch=5
    )
    print(cifar_trn_data.shape, cifar_tst_data.shape)

    # REDUCE DIMENSIONS
    if method_dim_red == "PCA" or method_dim_red == "pca":
        print("pca")
        reduced_trn_data, reduced_tst_data = pca_func(cifar_trn_data, cifar_tst_data)
    else:
        print("raw")
        reduced_trn_data, reduced_tst_data = cifar_trn_data, cifar_tst_data
    print(reduced_trn_data.shape, reduced_tst_data.shape)

    label_encoder = LabelEncoder()  # 将图片标签ID化
    label_encoder.fit_transform(cifar_tst_labels)

    print("start train")
    start = time()
    RomdomTrees = RondomForest(
        reduced_trn_data, cifar_trn_labels, 4, alpha="classification"
    )  # 4棵树,分类
    end = time()
    print("train time cost:", end - start)
    print("start test")
    
    print("---------------------RomdomTrees------------------------")
    yhat = predictTree(RomdomTrees, reduced_tst_data, alpha="classification")
    end2 = time()
    print("test time cost:", end2 - end)
    # print(cifar_tst_labels)
    # print(yhat)

    conf_mat, evalues = eval_model(cifar_tst_labels, yhat, label_encoder.classes_)
    print(conf_mat)  # 查看混淆矩阵
    plt.figure(figsize=(9, 7))
    sns.heatmap(conf_mat, annot=True, cmap="Oranges")
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.title("Confusion Matrix Heatmap")
    plt.show()

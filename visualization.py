from matplotlib.container import BarContainer
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union
import pandas as pd
import os


def load_font_config() -> None:
    config = {
        "font.family": 'serif',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    plt.rcParams.update(config)


def autolabel(rects:BarContainer, ax:plt.Axes, labels:List[any])->None:
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if isinstance(labels[i], float):
            name = "{:.3f}".format(labels[i])
        else:
            name = labels[i]
        ax.annotate(name, xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')


class visual(object):

    def __init__(self, ConMat: Union[List[List[int]], None] = None, labels: Union[List[any], None] = None, name: Union[str, None] = None) -> None:
        self.ConMat = pd.DataFrame(data=ConMat, index=labels, columns=labels, dtype=int)
        self.labels = labels
        self.name = name

    def getHeatMap(self, pathdir: str) -> None:
        if os.path.exists(pathdir) == False:
            os.mkdir(pathdir)
        fig, ax = plt.subplots()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        sns.heatmap(self.ConMat, annot=True, ax=ax, fmt="d")  # 画热力图
        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('predict')  # x 轴
        ax.set_ylabel('true')  # y 轴
        plt.savefig(f"{pathdir}/{self.name}.png")
        plt.close()

    def getPrecision(self, label: Union[any, None] = None) -> float:
        if label is not None:
            if label in self.labels:
                return self.ConMat[label][label] / self.ConMat[label].sum()
            else:
                raise ValueError("the label must in labels {}".format(self.labels))
        else:
            return sum([self.ConMat[i][i] / self.ConMat[i].sum() for i in self.labels]) / len(self.labels)

    def getRecall(self, label: Union[any, None] = None) -> float:
        if label is not None:
            if label in self.labels:
                return self.ConMat[label][label] / self.ConMat.loc[label].sum()
            else:
                raise ValueError("the label must in labels {}".format(self.labels))
        else:
            return sum([self.ConMat[i][i] / self.ConMat.loc[i].sum() for i in self.labels]) / len(self.labels)

    def getAcc(self) -> float:
        return sum([self.ConMat[i][i] for i in self.labels]) / self.ConMat.values.sum()

    def save(self, pathdir: str) -> None:
        if os.path.exists(pathdir) == False:
            os.mkdir(pathdir)
        self.ConMat.to_csv(f"{pathdir}/{self.name}.csv")

    def get(self, path: str) -> None:
        filename = os.path.basename(path)
        self.name = filename.split(".")[0]
        self.ConMat = pd.read_csv(path, index_col=0)
        self.labels = list(self.ConMat.index)

    def show(self, pathdir: str) -> None:
        if os.path.exists(pathdir) == False:
            os.mkdir(pathdir)
        Prec = [self.getPrecision(label) for label in self.labels]
        load_font_config()
        fig, ax = plt.subplots(ncols=2, figsize=(14, 8))
        ax[0].set_title(f"Precision and Recall of {self.name} for all classes")
        rects = ax[0].bar(self.labels, Prec)
        autolabel(rects, ax[0], Prec)
        Reca = [self.getRecall(label) for label in self.labels]
        ax[1].set_title(f"Recall of {self.name} for all classes")
        rects = ax[1].bar(self.labels, Reca)
        autolabel(rects, ax[1], Reca)
        plt.savefig(f"{pathdir}/{self.name}.png")
        plt.close()


def drawPrec(vislist: List[visual], path: str) -> None:
    pathdir = os.path.split(path)[0]
    if os.path.exists(pathdir) == False:
        os.mkdir(pathdir)
    dic = {vis.name: vis.getPrecision() for vis in vislist}
    fig, ax = plt.subplots()
    ax.set_title("mean Precision of all models")
    rects = ax.bar(dic.keys(), dic.values())
    autolabel(rects, ax, list(dic.values()))
    plt.savefig(path)
    plt.close()


def drawReca(vislist: List[visual], path: str) -> None:
    pathdir = os.path.split(path)[0]
    if os.path.exists(pathdir) == False:
        os.mkdir(pathdir)
    dic = {vis.name: vis.getRecall() for vis in vislist}
    fig, ax = plt.subplots()
    ax.set_title("mean Recall of all models")
    rects = ax.bar(dic.keys(), dic.values())
    autolabel(rects, ax, list(dic.values()))
    plt.savefig(path)
    plt.close()


def drawAcc(vislist: List[visual], path: str) -> None:
    pathdir = os.path.split(path)[0]
    if os.path.exists(pathdir) == False:
        os.mkdir(pathdir)
    dic = {vis.name: vis.getAcc() for vis in vislist}
    fig, ax = plt.subplots()
    ax.set_title("accruary of all models")
    rects = ax.bar(dic.keys(), dic.values())
    autolabel(rects, ax, list(dic.values()))
    plt.savefig(path)
    plt.close()


def drawFs(vislist: List[visual], path: str, beta: float = 1) -> None:
    pathdir = os.path.split(path)[0]
    if os.path.exists(pathdir) == False:
        os.mkdir(pathdir)
    dic = dict()
    for i in vislist:
        p, r = i.getPrecision(), i.getRecall()
        dic[i.name] = (1 + beta**2) * p * r / (beta**2 * p + r)
    fig, ax = plt.subplots()
    ax.set_title("F_Score of all models")
    rects = ax.bar(dic.keys(), dic.values())
    autolabel(rects, ax, list(dic.values()))
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    vis = visual()
    vis.get("result/LeNet.csv")
    print(vis.labels)
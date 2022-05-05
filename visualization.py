import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

def getHeatMap(CM:List[List[int]],labels:List[any],path:str) ->None:
    """
    对于给定样本标签的混淆矩阵，生成热度图并存于给定的path下
    """
    #temp = list(reversed(labels))
    temp =labels
    result = pd.DataFrame(data=CM,index=temp,columns=labels,dtype=int)
    print(result)
    sns.set()
    fig,ax = plt.subplots()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.heatmap(result, annot=True, ax=ax,fmt="d")  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x 轴
    ax.set_ylabel('true')  # y 轴
    plt.savefig(path)


if __name__ == "__main__":
    CM = [
[855,16,22,15,10,4,3,13,49,13],
[6,918,2,5,2,1,2,1,16,47],
[64,1,721,46,66,31,32,25,12,2],
[20,5,32,668,44,129,35,39,11,17],
[8,1,40,41,803,23,20,52,10,2],
[7,0,22,117,27,753,10,55,4,5],
[5,2,25,45,20,5,883,8,3,4],
[8,2,4,14,25,23,2,917,1,4],
[37,18,4,7,1,0,4,5,909,15],
[19,45,1,8,3,1,6,5,20,892]
]
    # CM = None
    # labels = ["飞机","汽车","鸟","猫","鹿","狗","蛙","马","羊","卡车"]
    labels = None
    getHeatMap(CM,labels,"test.png")

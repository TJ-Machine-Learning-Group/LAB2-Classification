import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

def getHeatMap(CM:List[List[int]],labels:List[any],path:str) ->None:
    #temp = list(reversed(labels))
    temp =labels
    result = pd.DataFrame(data=CM,index=temp,columns=labels,dtype=int)
    sns.set()
    fig,ax = plt.subplots()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.heatmap(result, annot=True, ax=ax,fmt="d")  # 画热力图
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x 轴
    ax.set_ylabel('true')  # y 轴
    plt.savefig(path)


if __name__ == "__main__":
    CM = [[626, 19, 87, 38, 48, 10, 10, 26, 87, 49], [23, 743, 9, 12, 7, 7, 11, 9, 45, 134], [52, 7, 567, 103, 113, 58, 35, 43, 9, 13], [19, 20, 61, 513, 97, 151, 60, 43, 14, 22],
          [20, 5, 88, 58, 656, 34, 60, 68, 7, 4], [8, 2, 58, 236, 79, 490, 23, 84, 11, 9], [10, 10, 57, 69, 53, 27, 743, 12, 8, 11], [10, 7, 46, 56, 65, 61, 11, 727, 2, 15],
          [61, 38, 25, 23, 17, 6, 9, 9, 764, 48], [25, 65, 16, 35, 12, 10, 21, 24, 32, 760]]
    labels = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "羊", "卡车"]
    getHeatMap(CM, labels, "result-png/test.png")

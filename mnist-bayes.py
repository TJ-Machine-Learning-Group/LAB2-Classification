import numpy as np
import struct

train_images_idx3_ubyte_file = 'data/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = 'data/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = 'data/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'data/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


if __name__ == '__main__':
    #读取训练数据
    train_images = load_train_images()
    train_labels = load_train_labels()


    #计算先验概率
    pre = [0] * 10
    count_pre = [0] * 10
    labels = train_labels.tolist()
    #print(np.int64(train_images[1].flatten() > 0))
    for i in range(60000):
        count_pre[int(labels[i])] = count_pre[int(labels[i])] + 1 #统计各类标签样本数量
    for i in range(10):
        pre[i] = (count_pre[i] + 1) / (60000 + 10)  #计算带平滑的先验概率
    # print(pre)


    # 计算类条件概率
    cond_1 = ([0.0] * 784) * 10
    for i in range(60000):
        temp = np.int64(train_images[i].flatten() > 24)    #当前样本特征值提取，并归一化为值为0或1
        cond_1[int(labels[i])] = cond_1[int(labels[i])] + temp   #统计各类标签样本的特征值

    for i in range(10):
        for j in range(784):
            cond_1[i][j] = (cond_1[i][j] + 1) / (count_pre[i] + 2)   #统计各类标签样本带平滑的类条件概率
    # print(cond_1[0])


    #计算后验概率
    #读取测试数据
    test_images = load_test_images()
    test_labels = load_test_labels()
    right_count = 0
    result = [[0 for i in range(11)] for i in range(11)]
    result[0] = [0, 0, 1, 2, 3, 4,5, 6, 7, 8, 9]
    for i in range(1,11):
        result[i][0] = i-1
    # print(result)
    for i in range(10000):
        maxvalue = 0
        max_type = -1
        temp = np.int64(test_images[i].flatten() > 0)  #当前测试样本特征值提取，并归一化为值为0或1
        for j in range(10):
            value = 1
            for t in range(784):
                if temp[t] == 1:
                    value = value * cond_1[j][t]
                else:
                    value = value * (1 - cond_1[j][t])
            value = value * pre[j]       #当前测试样本在j类标签上的后验概率(省略公式分母，只计算分子值)
            if value > maxvalue:
                maxvalue = value
                max_type = j        #选取当前测试样本最大的后验概率值所对应的类别，该类别即为测试样本的预测标签
        if max_type == int(test_labels[i]):
            right_count = right_count + 1
        # 比较并统计预测标签与真实标签
        result[int(test_labels[i]+1)][max_type+1] = result[int(test_labels[i]+1)][max_type+1] + 1
        #print("predict:", max_type)
        #print("true:", test_labels[i])
    print("Confusion Matrix:")
    for i in range(11):
        print(result[i])
    print("Correct Rate:", right_count/10000)


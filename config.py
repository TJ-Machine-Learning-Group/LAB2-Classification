import os
# 当前工作目录
CWD = "./"

label_num = 10
epoch = 40
learn_rate = 0.002
weight_decay = 0
model_savepath = os.path.join(CWD, "cnn_new.pth")
batch_size = 200
testdata_path = os.path.join(CWD, "mnist\\test")
traindata_path = os.path.join(CWD, "mnist\\train")

cifar10_train_path = os.path.join(CWD, 'cifar10\\train')
cifar10_test_path = os.path.join(CWD, 'cifar10\\test')
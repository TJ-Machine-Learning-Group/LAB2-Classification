import os
# 当前工作目录
CWD = "./"
model_savedir = "./model-results/"
label_num = 10
epoch = 50
hasdata=True
learn_rate = 0.002
weight_decay = 0
batch_size = 64
use_cuda = True
testdata_path = os.path.join(CWD, "mnist\\test")
traindata_path = os.path.join(CWD, "mnist\\train")

cifar10_train_path = os.path.join(CWD, 'cifar10\\train')
cifar10_test_path = os.path.join(CWD, 'cifar10\\test')

# -*- coding: utf-8 -*-
"""
Created on Fri May  6 20:56:01 2022

@author: I
"""
import torch
import numpy as np
import torch as tc
import pickle
import tarfile
import warnings
warnings.filterwarnings('ignore')

    
def convert_image_from_pack(raw_data):
    images = []
    labels = []
    
    def get_image_format(data):
        return np.array([[data[i], data[i + 1024], data[i + 2048]] for i in range(1024)])

    for i in range(len(raw_data[b'data'])):
        image_format = get_image_format(raw_data[b'data'][i])
        label = raw_data[b'labels'][i]
        #filepath = path + "/" + str(label) + "/" + raw_data['filenames'][i]
        #saveImage(image_format, "RGB", (32, 32), filepath)
        #logfile.write(filepath + "\t" + str(label) + "\n")
        images += [image_format]
        labels += [label]
        
    return np.array(images), labels
    
def load_cifar10(filename):
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    
    tar = tarfile.open(filename, "r:gz")
    
    tar_contents = [(file.name.split('/')[-1], file) for file in tar.getmembers()]
    tar_contents = sorted(filter(lambda x: x[0] in batches, tar_contents), key=lambda x: x[0])
    all_images = {}
    all_labels = {}

    for member in tar_contents:
        f = tar.extractfile(member[1])
        if f is not None:
            content = pickle.load(f, encoding='bytes')
            images, labels = convert_image_from_pack(content)
            all_images[member[0]] = images
            all_labels[member[0]] = labels
    tar.close()
    return all_images, all_labels

cfi, cfl= load_cifar10('cifar10\\train\\cifar-10-python.tar.gz')

def load_data():
    #img_train = np.concatenate([cfi['data_batch_1'], cfi['data_batch_1'], cfi['data_batch_3'],
    #    cfi['data_batch_4'], cfi['data_batch_5']])
    #label_train = np.concatenate([cfl['data_batch_1'], cfl['data_batch_1'], cfl['data_batch_3'],
    #    cfl['data_batch_4'], cfl['data_batch_5']])
    img_train = np.concatenate([cfi['data_batch_1'], cfi['data_batch_1']])
    label_train = np.concatenate([cfl['data_batch_1'], cfl['data_batch_1']])
    img_test = cfi['test_batch']
    label_test = cfl['test_batch']

    #  ??????????????????0-1??????
    img_train, img_test = tc.tensor(img_train/256, dtype=tc.float64, device='cuda:0'), tc.tensor(img_test/256, dtype=tc.float64, device='cuda:0')
    # ?????????????????????
    img_train, img_test = binarization(img_train), binarization(img_test)
    # ???????????????torch.long??????
    label_train, label_test = tc.tensor(label_train, dtype=tc.long, device='cuda:0'), tc.tensor(label_test, dtype=tc.long, device='cuda:0')
    return img_train, label_train, img_test, label_test


def binarization(picture_set):# ?????????????????????????????? ??????????????? >0.2?????? 1
    picture_set_out = (picture_set != 0).to(tc.float64)
    return picture_set_out


def Calculate_prior_and_conditional(img_train, lable_train):
    # ???????????????50000??????????????????????????????????????????????????????
    # ????????????????????????????????????????????????
    prior_probability = tc.zeros(10, dtype=tc.float64, device='cuda:0')
    conditional_probability = tc.zeros((10, 784, 2), dtype=tc.float64, device='cuda:0')
    for i in range(10):
        a = img_train[lable_train == i, :]
        prior_probability[i] = (lable_train == i).nonzero().shape[0]/lable_train.shape[0]
        # ????????????????????????0??????1???????????????
        conditional_probability[i, :, 0] = tc.div(a.shape[0] - tc.sum(a, dim=0), a.shape[0]).values + 1e-40
        conditional_probability[i, :, 1] = tc.div(tc.sum(a, dim=0), a.shape[0]).values + 1e-40

    return prior_probability, conditional_probability


def Calculate_probability(test_picture_set, conditional_probability, prior_probability):
	# ???????????????10000???????????????????????????
    probability = tc.zeros((10, test_picture_set.shape[0]), dtype=tc.float64, device='cuda:0')
    a = (test_picture_set == 0).to(tc.float64)
    b = (test_picture_set == 1).to(tc.float64)
    for i in range(10):
        class_conditional_probability = conditional_probability[i, :, :]
        a0_probability = tc.einsum('nl, l-> nl', [a, class_conditional_probability[:, 0]])
        a1_probability = tc.einsum('ab, b-> ab', [b, class_conditional_probability[:, 1]])
        a_probability = a0_probability + a1_probability
        probability[i, :] = tc.sum(tc.log(a_probability), dim=1) * prior_probability[i]
    return probability


def cul_accuracy(probability, lable_test):
    lable = tc.argmax(probability, dim=0)
    accuracy = ((lable == lable_test).nonzero().squeeze().shape[0]/lable.shape[0])
    return accuracy


if __name__ == '__main__':
	print('???????????????')
	img_train, lable_train, img_test, lable_test = load_data()
	print('????????????????????????????????????????????????')
	prior_probability, conditional_probability = Calculate_prior_and_conditional(img_train, lable_train)
	print('????????????')
	probability = Calculate_probability(img_test, conditional_probability, prior_probability)
	print('?????????????????????')
	accuracy = cul_accuracy(probability, lable_test)
	print('??????????????????????????????', accuracy)



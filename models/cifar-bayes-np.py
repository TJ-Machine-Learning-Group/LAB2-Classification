# coding: utf-8

import numpy as np
import pickle
import tarfile
import matplotlib.pyplot as plt
import scipy.stats as st
import itertools
import math
from sklearn.metrics import confusion_matrix as cmx
import warnings
warnings.filterwarnings('ignore')


def vectorize(X):
    return X.reshape((X.shape[0], np.prod(X.shape[1:])))

def dataSeparation(X, y, n_classes=10, arrToCol=True):
    if(arrToCol): X = vectorize(X)
    separatedData = [[] for i in range(n_classes)]
    for data, label in zip(X, y):
        separatedData[label].append(data)   
    return [np.array(data) for data in separatedData]
    
def convert_image_from_pack(raw_data):
    images = []
    labels = []
    
    def get_image_format(data):
        return np.array([[data[i], data[i + 1024], data[i + 2048]] for i in range(1024)])

    for i in range(len(raw_data[b'data'])):
        image_format = get_image_format(raw_data[b'data'][i])
        label = raw_data[b'labels'][i]
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

def plotLabelHist(y_test, y_train, title):
    fig, ax = plt.subplots(1,2,figsize=(13,5)); b=np.arange(0,11)-0.5
    h_t = ax[0].hist(y_test, edgecolor='gray', align='mid', bins=b )
    ax[0].set_xticks((np.arange(0,10))); ax[0].grid(axis='y', linestyle=':')
    ax[0].set_title("Distribution of labels in "+title+" Test")
    h_tr = ax[1].hist(y_train, edgecolor='gray', align='mid', bins=b )
    ax[1].set_xticks((np.arange(0,10))); ax[1].grid(axis='y', linestyle=':')
    ax[1].set_title("Distribution of labels in "+title+"Training")
    plt.show(); return h_t, h_tr

cfi, cfl= load_cifar10('cifar10\\train\\cifar-10-python.tar.gz')
cifar_X_train = np.concatenate([
    cfi['data_batch_1'], cfi['data_batch_1'], cfi['data_batch_3'],
    cfi['data_batch_4'], cfi['data_batch_5']])
cifar_y_train = np.concatenate([
    cfl['data_batch_1'], cfl['data_batch_1'], cfl['data_batch_3'],
    cfl['data_batch_4'], cfl['data_batch_5']])
cifar_X_test = cfi['test_batch']
cifar_y_test = cfl['test_batch']


# Sizes of CIFAR
print("Number of Training batches: 5")
print("Batch sizes: %d"% cifar_X_train.shape[0])
print("Image sizes: %d px with %d RGB colours each"%cifar_X_train.shape[1:])
cifar_h_t, cifar_h_tr = plotLabelHist(cifar_y_test, cifar_y_train, "CIFAR")
print("CIFAR Test -> [Min: %d, Max: %d, Mean: %d]"%
      (min(cifar_h_t[0]), max(cifar_h_t[0]), np.mean(cifar_h_t[0])))
print("CIFAR Training -> [Min: %d, Max: %d, Mean: %d]"%
      (min(cifar_h_tr[0]), max(cifar_h_tr[0]), np.mean(cifar_h_tr[0])))



def NBCDistributions(X, y, prior=None, sep=True):
    if(sep): X_sep = dataSeparation(X, y)
    else:    X_sep = X
    mean = [np.mean(x, axis=0) for x in X_sep]
    variance = [np.var(x, axis=0) for x in X_sep]
    #if(prior.all()==None):
    prior = np.histogram(y)[0]/len(y)
    return mean, variance, prior


def NBCPredict (X, mean, variance, prior,
                pdf=st.multivariate_normal.logpdf):
    ProbArray = []
    for m, v, p in zip(mean, variance, prior):
        ProbArray.append(np.log(p) + pdf(X, m, v, True))
    return np.argmax(np.array(ProbArray), axis=0),            np.max(np.array(ProbArray), axis=0)



def NBCEvaluate(pred, y):
    accuracy = np.sum(pred == y) / len(y)
    return accuracy


def NBCTrain (X_train, y_train, X_test, y_test,
              prior=None, step=5000, flag_print=True):
    
    TrainAcc = []
    TestAcc = []
    TestMeanLogL = []
    nEpochs = int(X_train.shape[0]/step)
    X_train = vectorize(X_train)
    X_test = vectorize(X_test)
    
    if(flag_print): print("Training a Naive Bayes Classifier")
    for i in range(1, nEpochs+1):
        
        m, v, prior = NBCDistributions(
            X_train[:(i*step)], y_train[:(i*step)],prior)
        
        trainPred, _        = NBCPredict (X_train[:(i*step)],m, v, prior)
        testPred, TestLLike = NBCPredict (X_test, m, v, prior)
        
        TrainAccuracy = NBCEvaluate(trainPred,  y_train[:(i*step)])
        TestAccuracy = NBCEvaluate(testPred, y_test)
        
        TrainAcc.append(TrainAccuracy*100)
        TestAcc.append(TestAccuracy*100)
        TestMeanLogL.append(np.mean(TestLLike))
        
        if(flag_print):
            print(" - Epoch %d:"        %(i*step),
                  "| TrainAcc = %.2f %%"%(TrainAcc[-1]),
                  "| TestAcc = %.2f %%" %(TestAcc[-1]),
                  "| TestLogLike = %.1f"%(TestMeanLogL[-1]))
        
    return TrainAcc, TestAcc, TestMeanLogL

def NBCPlotResults (TrainAcc, TestAcc, TestMeanLogL, title):
    fig, ax = plt.subplots(figsize=[7, 5])
    ax.plot(TrainAcc, label='Train Accuracy')
    ax.plot(TestAcc, label='Test Accuracy')
    ax2 = ax.twinx()
    ax2.plot(TestMeanLogL, ':', label='Test Log likelihood')
    fig.tight_layout()
    ax.set_title('Results of '+ title + 'training with NBC', fontsize=14)
    ax.set_xlabel('epoch(x1000)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylabel('TestLogLikelihood', fontsize=12)
    ax.legend(loc=9, fontsize=11)
    ax2.legend(fontsize=11)
    plt.show()



m, v, p = NBCDistributions(cifar_X_train, cifar_y_train)
cifar_yPred_test, _ = NBCPredict (vectorize(cifar_X_test), m, v, p)
cifar_TestAccuracy =  NBCEvaluate(cifar_yPred_test, cifar_y_test)
print("Test Accuracy on CIFAR for NBC: %.2f %%"%(cifar_TestAccuracy*100))
cifar_TrainAcc, cifar_TestAcc, cifar_TestLogLikelihood = NBCTrain(cifar_X_train, cifar_y_train, cifar_X_test, cifar_y_test)
NBCPlotResults(cifar_TrainAcc, cifar_TestAcc, cifar_TestLogLikelihood, 'CIFAR10')


def dataBalancing(X, y):
    X_sep = dataSeparation(X, y)
    n_min = min(np.histogram(y)[0])
    X_balanced = [k[:n_min] for k in X_sep]
    return X_balanced


def RGBtoGrayscale(X):
    return np.dot(X[...,:3], [0.299, 0.587, 0.114])


cifarGS_X_train = RGBtoGrayscale(cifar_X_train)
cifarGS_X_test = RGBtoGrayscale(cifar_X_test)
print("Old shape of CIFAR:", cifar_X_train.shape)
print("Grayscale shape of CIFAR", cifarGS_X_train.shape)

m, v, p = NBCDistributions(cifarGS_X_train, cifar_y_train)
cifarGS_yPred_test, _ = NBCPredict (vectorize(cifarGS_X_test), m, v, p)
cifarGS_TestAcc =  NBCEvaluate(cifarGS_yPred_test, cifar_y_test)
print("TestAcc on CIFAR grayscale for NBC: %.2f %%"%(cifarGS_TestAcc*100))


def plot_confusion_matrix(y, y_pred, title, classes=range(10)):
    cm = cmx(y, y_pred)
    plt.figure(figsize=(7,5))
    plt.imshow(cm, cmap=plt.cm.Purples)
    plt.title(title)
    plt.colorbar()
    plt.xticks(classes)
    plt.yticks(classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, math.ceil(cm[i, j]*100000)/100000,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > (2.*cm.max() / 3.) else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout(); 
    plt.show()


plot_confusion_matrix(cifar_y_test, cifar_yPred_test,
    'Confusion Matrix for CIFAR classified using a NBC', classes=range(10))


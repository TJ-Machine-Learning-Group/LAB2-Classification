import config
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import CNN_model
from tqdm import tqdm
from torch.autograd import Variable


def data_prepare():
    # 读取训练与测试数据
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.MNIST(config.traindata_path, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = datasets.MNIST(config.testdata_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, len(train_dataset), test_loader, len(test_dataset)


if __name__ == '__main__':
    epochs = config.epoch
    learn_rate = config.learn_rate
    weight_decay = config.weight_decay
    model_savepath = 'MNIST'+config.model_savepath

    # prepare data
    train_loader, train_count, test_loader, test_count = data_prepare()

    #loss func
    criterion = nn.CrossEntropyLoss()

    model = CNN_model.CNN()
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    confusion_matrix = [[0 for i in range(10)] for j in range(10)]

    #clean cuda
    torch.cuda.empty_cache()
    print('Start Train Process\n')
    for epoch in range(epochs):
        #init
        print('Epoch:', epoch+1)
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for i, data in tqdm(enumerate(train_loader, 1)):
            #use GPU
            img, label = data
            img = img.cuda()
            label = label.cuda()
            img = Variable(img)
            label = Variable(label)

            #forward
            result = model(img)
            loss = criterion(result, label)
            epoch_loss += loss.item() * label.size(0)

            #calculate
            _, pred = torch.max(result, 1)
            #print(pred)
            #print(label)
            correct = (pred == label).sum()
            epoch_accuracy += correct.item()
            if epoch == epochs-1:
                for j in range(config.batch_size):
                    confusion_matrix[label[j]][pred[j]] += 1

            #back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print
            print('{} epoch, {} batch, Loss: {:.6f}, Accuracy: {:.6f}'.format(
                epoch + 1, i, loss.item() * label.size(0) / (len(label)), correct.item() / (len(label))))

        #train result
        print('Finish {} epoch, Loss: {:.6f}, Accuracy: {:.6f}'.format(
            epoch + 1, epoch_loss / train_count, epoch_accuracy / train_count))

    print('Train Process End\n')
    print('Confusion Matrix\n')
    #for i in range(10):
    #    print(confusion_matrix[i])

    #save model
    torch.save(model.state_dict(), model_savepath)


    # Test Validate
    print('Start Test Validate Process\n')
    # init
    # use BN Layer, need .eval()
    model = CNN_model.CNN()
    model.load_state_dict(torch.load(model_savepath))
    model.eval()
    model = model.cuda()
    val_loss = 0.0
    val_accuracy = 0.0
    confusion_matrix0 = [[0 for i in range(10)] for j in range(10)]

    # clean cuda
    torch.cuda.empty_cache()
    print('Start Test Validate Process\n')
    for i, data in tqdm(enumerate(test_loader, 1)):
        # use GPU
        img, label = data
        with torch.no_grad():
            img = Variable(img).cuda()
        with torch.no_grad():
            label = Variable(label).cuda()

        # forward
        result = model(img)
        loss = criterion(result, label)
        val_loss += loss.item() * label.size(0)

        # calculate
        _, pred = torch.max(result, 1)
        #print(pred)
        #print(label)
        correct = (pred == label).sum()
        val_accuracy += correct.item()

        for j in range(config.batch_size):
                confusion_matrix0[label[j]][pred[j]] += 1
        # print
        print('Validate: {} batch, Loss: {:.6f}, Accuracy: {:.6f}'.format(
            i, loss.item() * label.size(0) / (len(label)), correct.item() / (len(label))))

    # test result
    print('Finish Validate, Loss: {:.6f}, Accuracy: {:.6f}'.format(
        val_loss / test_count, val_accuracy / test_count))

    print('Validate Test Validate End\n')
    print('Confusion Matrix\n')
    for i in range(10):
        print(confusion_matrix0[i])


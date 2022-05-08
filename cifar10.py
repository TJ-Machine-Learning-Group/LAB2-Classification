import config
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
from torch.autograd import Variable
import ssl

from models import *
import visualization
def prepare_cifar10():
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR10(config.cifar10_train_path, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(config.cifar10_test_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, len(train_dataset), test_loader, len(test_dataset)


def Test(model, criterion, test_loader, test_count, model_savepath: str):
    # Test Validate
    print('Start Test Validate Process\n')
    # init
    # use BN Layer, need .eval()

    model.load_state_dict(torch.load(model_savepath))
    model.eval()
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
            if config.use_cuda == True:
                img = Variable(img).cuda()
        with torch.no_grad():
            if config.use_cuda == True:
                label = Variable(label).cuda()

        # forward
        result = model(img)
        loss = criterion(result, label)
        val_loss += loss.item() * label.size(0)

        # calculate
        _, pred = torch.max(result, 1)
        # print(pred)
        # print(label)
        correct = (pred == label).sum()
        val_accuracy += correct.item()

        for j in range(img.shape[0]):
            confusion_matrix0[label[j]][pred[j]] += 1
        # print
        #print('Validate: {} batch, Loss: {:.6f}, Accuracy: {:.6f}'.format(i, loss.item() * label.size(0) / (len(label)), correct.item() / (len(label))))

    # test result
    print('Finish Validate, Loss: {:.6f}, Accuracy: {:.6f}'.format(val_loss / test_count, val_accuracy / test_count))

    print('Validate Test Validate End\n')
    print('Confusion Matrix\n')
    for i in range(10):
        print(confusion_matrix0[i])


def Train(model, criterion, train_loader, train_count, model_savepath: str):
    epochs = config.epoch
    learn_rate = config.learn_rate
    weight_decay = config.weight_decay

    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    confusion_matrix = [[0 for i in range(10)] for j in range(10)]

    #clean cuda
    torch.cuda.empty_cache()
    print('Start Train Process\n')
    for epoch in range(epochs):
        #init
        print('Epoch:', epoch + 1)
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for i, data in tqdm(enumerate(train_loader, 1)):
            #use GPU

            img, label = data

            if config.use_cuda == True:
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
            # print(pred)
            # print(label)
            correct = (pred == label).sum()
            epoch_accuracy += correct.item()
            if epoch == epochs - 1:
                for j in range(img.shape[0]):
                    confusion_matrix[label[j]][pred[j]] += 1

            #back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # #print
            # print('{} epoch, {} batch, Loss: {:.6f}, Accuracy: {:.6f}'.format(
            #     epoch + 1, i, loss.item() * label.size(0) / (len(label)), correct.item() / (len(label))))

        #train result
        print('Finish {} epoch, Loss: {:.6f}, Accuracy: {:.6f}'.format(epoch + 1, epoch_loss / train_count, epoch_accuracy / train_count))

        #save model
        torch.save(model.state_dict(), model_savepath)  #防止等太久 先存下来

    print('Train Process End\n')
    print('Confusion Matrix\n')
    for i in range(10):
        print(confusion_matrix[i])

    #save model
    torch.save(model.state_dict(), model_savepath)


if __name__ == '__main__':
    models = [
        sample_CNN(),
        #DenseNet121(),  #out of cuda memory
        #DPN26(),#out of cuda memory
        #EfficientNetB0(),  #out of cuda memory
        #GoogLeNet(),#out of cuda memory
        LeNet(),
        MobileNet(),
        MobileNetV2(),
        #DPN92(),  #out of cuda memory
        SENet18(),
        ShuffleNetV2(1)
    ]
    # prepare data
    train_loader, train_count, test_loader, test_count = prepare_cifar10()
    vislist = list()
    #prepare model
    for model in models:
        if config.hasdata == False:
            if config.use_cuda == True:
                model = model.cuda()  #gpu
            model_savepath = config.model_savedir + model.name + ".pth"

            #prepare loss func
            criterion = nn.CrossEntropyLoss()
            # NOTE 这里注释掉训练部分, 要训练的记得取消这里的注释
            #Train(model, criterion, train_loader, train_count, model_savepath)
            visCM = Test(model, criterion, test_loader, test_count, model_savepath)
            vis = visualization.visual(visCM, config.labels, model.name)
            # vis.getHeatMap("heatmap/{}.png".format(model.name))
            vis.save("result")
        else:
            vis = visualization.visual()
            vis.get(f"result/{model.name}.csv")
        vis.show("result/analyze")
        vis.getHeatMap("result/heatmap")
        vislist.append(vis)
    visualization.drawPrec(vislist, "result/Prec.png")
    visualization.drawReca(vislist, "result/Reca.png")
    visualization.drawAcc(vislist, "result/Acc.png")
    visualization.drawFs(vislist, "result/Fs.png")

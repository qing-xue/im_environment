import torch
from torch.autograd import Variable
import time
import utils
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def visualize_model(vgg, dataloader, num_images=6, class_names=None):
    """预测结果可视化

    参数说明：
        dataloaders: 载入测试数据
    """
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        size = inputs.size()[0]

        inputs, labels = Variable(inputs.to(device), volatile=True), Variable(labels.to(device), volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        utils.show_databatch(inputs.data.cpu(), labels.data.cpu(), class_names)
        print("Prediction:")
        utils.show_databatch(inputs.data.cpu(), predicted_labels, class_names)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training)  # Revert model back to original training state


def eval_model(vgg, criterion, dataloader):
    """在测试集上评估模型

    参数说明：
        dataloaders: 载入测试数据
    """
    since = time.time()
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloader)
    num_samples = len(dataloader.dataset)  # ugly
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloader):
        if i % 10 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data  # data[0] for GPU?
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test / num_samples
    avg_acc = acc_test / num_samples
        
    elapsed_time = time.time() - since
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def train_model(dataloader, vgg, criterion, optimizer, num_epochs=10):
    """训练模型

    参数说明：
        dataloaders: 载入训练
        vgg: 预训练模型
        criterion: 计算损失
        optimizer: 优化器
    """
    since = time.time()
    train_batches = len(dataloader)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        acc_train = 0
        
        vgg.train(True)
                
        for i, data in enumerate(dataloader):
            if i % 10 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                
            inputs, labels = data
            
            # if use_gpu:
            #     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # else:
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            optimizer.zero_grad()     
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.data  # data[0] for GPU?
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / train_batches
        avg_acc = acc_train / train_batches

        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print('-' * 10)
        
    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return vgg


def vgg_customize(vgg16, out):
    """替换 vgg 网络最后一层的输出向量

    Params:
        out: 输出向量个数
    """
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, out)])
    vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

    print(vgg16)
    return vgg16

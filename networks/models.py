import torch
from torch.autograd import Variable
import time
import copy
import utils


def visualize_model(vgg, dataloaders, num_images=6, use_gpu=True):
    """预测结果可视化

    参数说明：
        dataloaders: 载入测试数据 ['test'] or ['val']
    """
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        size = inputs.size()[0]
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        utils.show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        utils.show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training)  # Revert model back to original training state


def eval_model(vgg, criterion, dataloaders, use_gpu=True):
    """在测试集上评估模型

    参数说明：
        dataloaders: 载入测试数据 ['test'] or ['val']
    """
    since = time.time()
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders['val'])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders['val']):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data  # data[0] for GPU?
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    # avg_loss = loss_test / dataset_sizes[TEST]
    avg_loss = loss_test / test_batches
    # avg_acc = acc_test / dataset_sizes[TEST]
    avg_acc = acc_test / test_batches
        
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def train_model(dataloaders, vgg, criterion, optimizer, num_epochs=10, use_gpu=False):
    """训练模型

    参数说明：
        dataloaders: 载入训练、验证数据 ['train', 'val']
        vgg: 预训练模型
        criterion: 计算损失
        optimizer: 优化器
    """
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
                
        for i, data in enumerate(dataloaders['train']):
            if i % 10 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
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
        
        print()
        # * 2 as we only used half of the dataset
        # avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_loss = loss_train / train_batches
        # avg_acc = acc_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train / train_batches
        
        vgg.train(False)
        vgg.eval()
            
        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                        
            optimizer.zero_grad()
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.data  # data[0] for GPU?
            acc_val += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        # avg_loss_val = loss_val / dataset_sizes[VAL]
        avg_loss_val = loss_val / val_batches
        # avg_acc_val = acc_val / dataset_sizes[VAL]
        avg_acc_val = acc_val / val_batches
                
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
        
    vgg.load_state_dict(best_model_wts)
    return vgg

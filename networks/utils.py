import os
import matplotlib.pyplot as plt
import torchvision
plt.ion()  


def mkdir(floder):
    if not os.path.exists(floder):
        os.makedirs(floder)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes, class_names=None):
    if not class_names:
        class_names = classes
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


if __name__ = '__main__':
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders[TRAIN]))
    utils.show_databatch(inputs, classes, class_names)

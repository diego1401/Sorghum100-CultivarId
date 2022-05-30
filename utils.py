import torch.nn as nn
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import os

# Some Azure kernels do not have and cannot install plt
try:
    import matplotlib.pyplot as plt
except:
    pass


def accuracy(net, test_loader, cuda=True):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data['image'], data['target']
            if cuda:
                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    acc = correct / total
    print(f'Accuracy of the network on the test images:{(100 * acc)}')
    return correct / total


def train(net, optimizer, criterion,
          train_loader, test_loader, modelname,
          n_epoch=5, train_acc_period=100, cuda=True,
          output_file=None):
    train_acc = []
    test_acc = []

    save_progress(modelname, net, [0], [0])

    for epoch in tqdm(range(n_epoch), file=output_file):
        if output_file is not None:
            output_file.write("\n")
        total = 0
        correct = 0
        tmp_train = []
        for i, data in enumerate(train_loader, 0):
            net.train()
            inputs, labels = data['image'], data['target']
            if cuda:
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if i % train_acc_period == train_acc_period - 1:
                acc = correct / total
                train_acc.append(acc)
                tmp_train.append(acc)
                console_output_train = '[%d, %5d] acc: %.3f' % (epoch + 1, i + 1, acc)
                print(console_output_train)
                if output_file is not None:
                    output_file.write(console_output_train)
                    output_file.write("\n")
                    output_file.flush()
                # print()

        test_accuracy = accuracy(net, test_loader, cuda=cuda)
        test_acc.append(test_accuracy)
        console_output_test = '[%d] acc: %.3f' % (epoch + 1, test_accuracy)
        if output_file is not None:
            output_file.write(console_output_test)
            output_file.write("\n")
            output_file.flush()
        save_progress(modelname, net, tmp_train, [test_acc[-1]])
        if output_file is not None:
            output_file.write("Saving Progress")
            output_file.write("\n")
            output_file.flush()
        # print('[%d] acc: %.3f' %(epoch + 1, test_accuracy))

    print('Finished Training')
    return train_acc, test_acc


def save_progress(modelname, network, train_acc_netPretrained, test_acc_netPretrained):
    PATH = f'models/{modelname}'
    torch.save(network, PATH)

    if os.path.isfile(f'plots/{modelname}.npy'):
        with open(f'plots/{modelname}.npy', 'rb') as f:
            train_acc = np.load(f)
            test_acc = np.load(f)
        train_acc_netPretrained = list(train_acc) + train_acc_netPretrained
        test_acc_netPretrained = list(test_acc) + test_acc_netPretrained
    with open(f'plots/{modelname}.npy', 'wb') as f:
        np.save(f, train_acc_netPretrained)
        np.save(f, test_acc_netPretrained)


# random choice class from
# https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image

import random


class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)


#########################################################################################


def apply_clahe(image):
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.split(image)
    image = list(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image[2] = clahe.apply(image[2])
    image = cv2.merge(image)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image)


def plot_from_loader(loader, title="", save=False):
    # to plot examples of the dataset
    for i, data in enumerate(loader, 0):
        v_slice = data["image"]
        # subplot(r,c) provide the no. of rows and columns
        n_rows = 4
        f, axarr = plt.subplots(n_rows, 5, figsize=(20, 15))

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for i in range(n_rows):
            for k in range(5):
                image = v_slice[i * 5 + k].permute(1, 2, 0)
                image = torch.clip(image, 0, 1)
                axarr[i][k].imshow(image)
                axarr[i][k].axis("off")

        plt.tight_layout()
        if save:
            if title == "":
                raise ValueError("title is required when saving")
            plt.savefig("figures/" + title)
        plt.show()
        break

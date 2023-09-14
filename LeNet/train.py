import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    # batch_size --批量大小
    # shuffle --是否将图片打乱
    # num_workers --windows系统只能设置为0，Linux系统可以自己设置
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # 查看图片时batch_size设置小点，一次查看的数量
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    # 标签对应相应的 index 值，从 0 开始
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # 下面这个展示图片的代码不能单独放在外面要和上面在同一作用域下
    # # print labels
    # print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(4)))
    # # show images 调用函数imshow
    # imshow(torchvision.utils.make_grid(val_image))

# 想查看图片时要取消下面注释
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # print labels-----下面这是错误的放置位置，会导致参数无法传递
# print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(val_image))


    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            # input [36, 3 ,W ,H]--batch size 36  --channel 3
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        # ----------TODO------------
        # define a network 
        # 32 * 32 * 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 1, 1),
            # 32 * 32 * 96
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(2, 2),
            # 16 * 16 * 96
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 1, 1),
            # 16 * 16 * 256
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(2, 2),
            # 8 * 8 * 256
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            # 8 * 8 * 384
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            # 8 * 8 * 384
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            # 8 * 8 * 256
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 4 * 4 * 256
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            # 4096 * 1
            nn.Linear(4096, 256),
            # 256 * 1
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 10),
            # 10 * 1
        )
        # ----------TODO------------

    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.dense(x)
        # ----------TODO------------

        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 

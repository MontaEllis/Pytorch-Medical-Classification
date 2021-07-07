import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(num_class):
    return VGG(make_layers(cfg['A'], batch_norm=True),num_class=num_class)

def vgg13_bn(num_class):
    return VGG(make_layers(cfg['B'], batch_norm=True),num_class=num_class)

def vgg16_bn(num_class):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_class=num_class)

def vgg19_bn(num_class):
    return VGG(make_layers(cfg['E'], batch_norm=True),num_class=num_class)


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 224
    x = torch.Tensor(1, 1, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))
    
    model = vgg16_bn(num_class=2).to(device)
    

    out1 = model(x)
    print("out size: {}".format(out1.size()))
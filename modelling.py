import torch
import torchvision
from torch import nn
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet152_Weights


class FashionResnet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        if hparams['resnet_type'] == 'resnet101':
            self.model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif hparams['resnet_type'] == 'resnet50':
            self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = torchvision.models.resnet152(weights=ResNet152_Weights.DEFAULT)

        self.model.fc = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)


class FashionClassifictions(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.gender_linear1 = nn.Linear(hparams['layer1'], hparams['layer2'])
        self.gender_linear2 = nn.Linear(hparams['layer2'], hparams['layer3'])
        self.gender_out = nn.Linear(hparams['layer3'], hparams['gender_classes'])

        self.mastercat_linear1 = nn.Linear(hparams['layer1'], hparams['layer2'])
        self.mastercat_linear2 = nn.Linear(hparams['layer2'], hparams['layer3'])
        self.mastercat_out = nn.Linear(hparams['layer3'], hparams['mastercat_classes'])

        self.subcat_linear1 = nn.Linear(hparams['layer1'], hparams['layer2'])
        self.subcat_linear2 = nn.Linear(hparams['layer2'], hparams['layer3'])
        self.subcat_out = nn.Linear(hparams['layer3'], hparams['subcat_classes'])

        self.color_linear1 = nn.Linear(hparams['layer1'], hparams['layer2'])
        self.color_linear2 = nn.Linear(hparams['layer2'], hparams['layer3'])
        self.color_out = nn.Linear(hparams['layer3'], hparams['color_classes'])

        if hparams['activation'] == 'ReLU':
            self.activation = nn.ReLU()
        elif hparams['activation'] == 'gelu':
            self['activation'] = nn.GELU()
        self.dropout = nn.Dropout(hparams['dropout_val'])

    def forward(self, out):
        gender_out = self.activation(self.dropout((self.gender_linear1(out))))
        gender_out = self.activation(self.dropout(self.gender_linear2(gender_out)))
        gender_out = self.gender_out(gender_out)

        master_out = self.activation(self.dropout((self.mastercat_linear1(out))))
        master_out = self.activation(self.dropout(self.mastercat_linear2(master_out)))
        master_out = self.mastercat_out(master_out)

        subcat_out = self.activation(self.dropout((self.subcat_linear1(out))))
        subcat_out = self.activation(self.dropout(self.subcat_linear2(subcat_out)))
        subcat_out = self.subcat_out(subcat_out)

        color_out = self.activation(self.dropout((self.color_linear1(out))))
        color_out = self.activation(self.dropout(self.color_linear2(color_out)))
        color_out = self.color_out(color_out)

        return gender_out, master_out, subcat_out, color_out


def test_modelling():
    hparams = {'path': 'data/fashion-dataset/images/', 'batch_size': 64, 'gender_classes': 5, 'mastercat_classes': 4,
               'subcat_classes': 32, 'color_classes': 44, 'resnet_type': 'resnet101', 'layer1': 2048, 'layer2': 1024,
               'layer3': 256, 'activation': 'ReLU', 'dropout_val': 0.3}
    sample_resnet = FashionResnet(hparams)
    sample_data = torch.randn((16, 3, 256, 256))
    print(sample_data.shape)
    out = sample_resnet(sample_data)
    print("The output shape is :", out.shape)
    sample_linears = FashionClassifictions(hparams)
    gender_out, master_out, sub_out, color_out = sample_linears(out)
    print(gender_out.shape, master_out.shape, sub_out.shape, color_out.shape)

if __name__ == '__main__':
    test_modelling()
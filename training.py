import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from modelling import FashionResnet, FashionClassifictions


class FashionPrediction(pl.LightningModule):
    """
    Class to facilitate how training will happen

    Attrubutes:
    hparams (dict) : The hyper parameters.
    master_weights, subcat_weights, color_weights, gender_weights (tensor): loss weightage for the attributes for
    training only
    """
    def __init__(self, hparams, master_weights, subcat_weights, color_weights, gender_weights):
        super().__init__()
        torch.set_printoptions(profile='full', linewidth=400)
        self.save_hyperparameters()
        self.feature_model = FashionResnet(self.hparams['hparams'])
        self.classification_model = FashionClassifictions(self.hparams['hparams'])
        self.gender_acc = Accuracy(task='multiclass', num_classes=self.hparams['hparams']['gender_classes'])
        self.master_acc = Accuracy(task='multiclass', num_classes=self.hparams['hparams']['mastercat_classes'])
        self.subcat_acc = Accuracy(task='multiclass', num_classes=self.hparams['hparams']['subcat_classes'])
        self.color_acc = Accuracy(task='multiclass', num_classes=self.hparams['hparams']['color_classes'])
        self.gender_f1 = F1Score(task='multiclass', num_classes=self.hparams['hparams']['gender_classes'], average='macro')
        self.mastercat_f1 = F1Score(task='multiclass', num_classes=self.hparams['hparams']['mastercat_classes'], average='macro')
        self.subcat_f1 = F1Score(task='multiclass', num_classes=self.hparams['hparams']['subcat_classes'], average='macro')
        self.color_f1 = F1Score(task='multiclass', num_classes=self.hparams['hparams']['color_classes'], average='macro')
        self.gender_confusion = ConfusionMatrix(task='multiclass', num_classes=self.hparams['hparams']['gender_classes'])
        self.mastercat_confusion = ConfusionMatrix(task='multiclass', num_classes=self.hparams['hparams']['mastercat_classes'])
        self.subcat_confusion = ConfusionMatrix(task='multiclass', num_classes=self.hparams['hparams']['subcat_classes'])
        self.color_confusion = ConfusionMatrix(task='multiclass', num_classes=self.hparams['hparams']['color_classes'])
        self.gender_label = ['Boys', 'Girls', 'Men', 'Unisex', 'Women']
        self.master_label = ['Accessories', 'Apparel', 'Footwear', 'Personal Care']
        self.subcat_label = ['Accessories', 'Apparel Set', 'Bags', 'Belts', 'Bottomwear', 'Cufflinks', 'Dress', 'Eyes',
                             'Eyewear', 'Flip Flops', 'Fragrance', 'Headwear', 'Innerwear', 'Jewellery', 'Lips',
                             'Loungewear and Nightwear', 'Makeup', 'Mufflers', 'Nails', 'Sandal', 'Saree', 'Scarves',
                             'Shoe Accessories', 'Shoes', 'Skin', 'Skin Care', 'Socks', 'Stoles', 'Ties', 'Topwear',
                             'Wallets', 'Watches']
        self.color_label = ['Beige', 'Black', 'Blue', 'Bronze', 'Brown', 'Burgundy', 'Charcoal', 'Coffee Brown',
                            'Copper', 'Cream', 'Gold', 'Green', 'Grey', 'Grey Melange', 'Khaki', 'Lavender', 'Magenta',
                            'Maroon', 'Mauve', 'Metallic', 'Multi', 'Mushroom Brown', 'Mustard', 'Navy Blue', 'Nude',
                            'Off White', 'Olive', 'Orange', 'Peach', 'Pink', 'Purple', 'Red', 'Rose', 'Rust',
                            'Sea Green', 'Silver', 'Skin', 'Steel', 'Tan', 'Taupe', 'Teal', 'Turquoise Blue',
                            'White', 'Yellow']
        self.color_weights = torch.tensor(self.hparams['color_weights'])
        self.master_weights = torch.tensor(self.hparams['master_weights'])
        self.subcat_weights = torch.tensor(self.hparams['subcat_weights'])
        self.gender_weights = torch.tensor(self.hparams['gender_weights'])
        self.master_criterion = nn.CrossEntropyLoss(weight=self.master_weights)
        self.subcat_criterion = nn.CrossEntropyLoss(weight=self.subcat_weights)
        self.gender_criterion = nn.CrossEntropyLoss(weight=self.gender_weights)
        self.color_criterion = nn.CrossEntropyLoss(weight=self.color_weights)


    def forward(self, image, transforms):
        pass

    def training_step(self, input_batch, batch_idx):
        image_tensors = input_batch['transformed_image']
        gender_targets = input_batch['gender']
        master_targets = input_batch['master_category']
        sub_targets = input_batch['sub_category']
        color_targets = input_batch['base_color']

        features = self.feature_model(image_tensors)
        gender_logits, master_logits, sub_logits, color_logits = self.classification_model(features)
        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        master_loss = self.master_criterion(master_logits, master_targets)
        sub_loss = self.subcat_criterion(sub_logits, sub_targets)
        color_loss = self.color_criterion(color_logits, color_targets)

        total_loss = gender_loss + master_loss + sub_loss + color_loss

        gender_predict = torch.argmax(gender_logits, dim=1)
        master_predict = torch.argmax(master_logits, dim=1)
        sub_predict = torch.argmax(sub_logits, dim=1)
        color_predict = torch.argmax(color_logits, dim=1)

        gender_acc = self.gender_acc(gender_predict, gender_targets)
        master_acc = self.master_acc(master_predict, master_targets)
        sub_acc = self.subcat_acc(sub_predict, sub_targets)
        color_acc = self.color_acc(color_predict, color_targets)

        total_acc = (gender_acc + master_acc + sub_acc + color_acc) / 4

        self.log_dict({
            'gender-loss': gender_loss, 'master-loss': master_loss, 'sub-loss': sub_loss, 'color-loss': color_loss,
            'gender-acc': gender_acc, 'total-loss': total_loss, 'master-acc': master_acc, 'sub-acc': sub_acc,
            'color-acc': color_acc, 'total-acc': total_acc
        }, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, input_batch, batch_idx):
        image_tensors = input_batch['transformed_image']
        gender_targets = input_batch['gender']
        master_targets = input_batch['master_category']
        sub_targets = input_batch['sub_category']
        color_targets = input_batch['base_color']

        features = self.feature_model(image_tensors)
        gender_logits, master_logits, sub_logits, color_logits = self.classification_model(features)
        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        master_loss = self.master_criterion(master_logits, master_targets)
        sub_loss = self.subcat_criterion(sub_logits, sub_targets)
        color_loss = self.color_criterion(color_logits, color_targets)

        total_loss = gender_loss + master_loss + sub_loss + color_loss

        gender_predict = torch.argmax(gender_logits, dim=1)
        master_predict = torch.argmax(master_logits, dim=1)
        sub_predict = torch.argmax(sub_logits, dim=1)
        color_predict = torch.argmax(color_logits, dim=1)

        gender_acc = self.gender_acc(gender_predict, gender_targets)
        master_acc = self.master_acc(master_predict, master_targets)
        sub_acc = self.subcat_acc(sub_predict, sub_targets)
        color_acc = self.color_acc(color_predict, color_targets)

        total_acc = (gender_acc + master_acc + sub_acc + color_acc) / 4

        self.log_dict({
            'val-gender-loss': gender_loss, 'val-master-loss': master_loss, 'val-sub-loss': sub_loss, 'val-color-loss': color_loss,
            'val-gender-acc': gender_acc, 'val-master-acc': master_acc, 'val-sub-acc': sub_acc, 'val-color-acc': color_acc,
            'val-total-acc': total_acc, 'val-total-loss': total_loss
        }, on_step=False, on_epoch=True, prog_bar=False)

        val_dict = {
            'gender_predict': gender_predict,
            'gender_targets': gender_targets,
            'master_predict': master_predict,
            'master_targets': master_targets,
            'sub_predict': sub_predict,
            'sub_targets': sub_targets,
            'color_predict': color_predict,
            'color_targets': color_targets
        }

        return val_dict

    def validation_epoch_end(self, validation_outputs):
        gender_preds, gender_targets, master_preds, master_targets = [], [], [], []
        sub_preds, sub_targets, color_preds, color_targets = [], [], [], []
        for x in validation_outputs:
            gender_preds.append(x['gender_predict'])
            gender_targets.append(x['gender_targets'])
            master_preds.append(x['master_predict'])
            master_targets.append(x['master_targets'])
            sub_preds.append(x['sub_predict'])
            sub_targets.append(x['sub_targets'])
            color_preds.append(x['color_predict'])
            color_targets.append(x['color_targets'])

        all_gender_preds = torch.stack(gender_preds[0:-1]).view(-1)
        all_gender_targets = torch.stack(gender_targets[0:-1]).view(-1)

        all_master_preds = torch.stack(master_preds[0:-1]).view(-1)
        all_master_targets = torch.stack(master_targets[0:-1]).view(-1)

        all_sub_preds = torch.stack(sub_preds[0:-1]).view(-1)
        all_sub_targets = torch.stack(sub_targets[0:-1]).view(-1)

        all_color_preds = torch.stack(color_preds[0:-1]).view(-1)
        all_color_targets = torch.stack(color_targets[0:-1]).view(-1)

        all_gender_preds = torch.cat((all_gender_preds, gender_preds[-1:][0]))
        all_gender_targets = torch.cat((all_gender_targets, gender_targets[-1:][0]))

        all_master_preds = torch.cat((all_master_preds, master_preds[-1:][0]))
        all_master_targets = torch.cat((all_master_targets, master_targets[-1:][0]))

        all_sub_preds = torch.cat((all_sub_preds, sub_preds[-1:][0]))
        all_sub_targets = torch.cat((all_sub_targets, sub_targets[-1:][0]))

        all_color_preds = torch.cat((all_color_preds, color_preds[-1:][0]))
        all_color_targets = torch.cat((all_color_targets, color_targets[-1:][0]))

        gender_confusion_metric = self.gender_confusion(all_gender_preds, all_gender_targets)
        master_confusion_metric = self.mastercat_confusion(all_master_preds, all_master_targets)
        sub_confusion_metric = self.subcat_confusion(all_sub_preds, all_sub_targets)
        color_confusion_metric = self.color_confusion(all_color_preds, all_color_targets)

        gender_f1_score = self.gender_f1(all_gender_preds, all_gender_targets)
        master_f1_score = self.mastercat_f1(all_master_preds, all_master_targets)
        sub_f1_score = self.subcat_f1(all_sub_preds, all_sub_targets)
        color_f1_score = self.color_f1(all_color_preds, all_color_targets)

        total_f1_score = (gender_f1_score + master_f1_score + sub_f1_score + color_f1_score)/4

        self.log_dict({
            'gender-f1score': gender_f1_score, 'master-f1score': master_f1_score, 'sub-f1score': sub_f1_score,
            'color-f1score': color_f1_score, 'total-f1score': total_f1_score
        })

        print("gender_confusion_metric: \n", gender_confusion_metric)
        print("master_confusion_metric: \n", master_confusion_metric)
        print("sub_confusion_metric: \n", sub_confusion_metric)
        print("color_confusion_metric: \n", color_confusion_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['hparams']['lr'])
        return optimizer


if __name__ == '__main__':
    pass


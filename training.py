import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from modelling import FashionResnet, FashionClassifictions


class FashionPrediction(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.feature_model = FashionResnet(self.hparams['hparams'])
        self.classification_model = FashionClassifictions(self.hparams['hparams'])
        self.acc = Accuracy()
        self.criterion = nn.CrossEntropyLoss()
        self.gender_f1 = F1Score(self.hparams['hparams']['gender_classes'], average='macro', mdmc_average='global')
        self.mastercat_f1 = F1Score(self.hparams['hparams']['mastercat_classes'], average='macro', mdmc_average='global')
        self.subcat_f1 = F1Score(self.hparams['hparams']['subcat_classes'], average='macro', mdmc_average='global')
        self.color_f1 = F1Score(self.hparams['hparams']['color_classes'], average='macro', mdmc_average='global')
        self.gender_confusion = ConfusionMatrix(self.hparams['hparams']['gender_classes'])
        self.mastercat_confusion = ConfusionMatrix(self.hparams['hparams']['mastercat_classes'])
        self.subcat_confusion = ConfusionMatrix(self.hparams['hparams']['subcat_classes'])
        self.color_confusion = ConfusionMatrix(self.hparams['hparams']['color_classes'])
        self.softmax = nn.Softmax(dim=1)

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
        gender_loss = self.criterion(gender_logits, gender_targets)
        master_loss = self.criterion(master_logits, master_targets)
        sub_loss = self.criterion(sub_logits, sub_targets)
        color_loss = self.criterion(color_logits, color_targets)

        total_loss = gender_loss + master_loss + sub_loss + color_loss

        gender_predict = torch.argmax(gender_logits, dim=1)
        master_predict = torch.argmax(master_logits, dim=1)
        sub_predict = torch.argmax(sub_logits, dim=1)
        color_predict = torch.argmax(color_logits, dim=1)

        gender_acc = self.acc(gender_predict, gender_targets)
        master_acc = self.acc(master_predict, master_targets)
        sub_acc = self.acc(sub_predict, sub_targets)
        color_acc = self.acc(color_predict, color_targets)

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
        gender_loss = self.criterion(gender_logits, gender_targets)
        master_loss = self.criterion(master_logits, master_targets)
        sub_loss = self.criterion(sub_logits, sub_targets)
        color_loss = self.criterion(color_logits, color_targets)

        total_loss = gender_loss + master_loss + sub_loss + color_loss

        gender_predict = torch.argmax(gender_logits, dim=1)
        master_predict = torch.argmax(master_logits, dim=1)
        sub_predict = torch.argmax(sub_logits, dim=1)
        color_predict = torch.argmax(color_logits, dim=1)

        gender_acc = self.acc(gender_predict, gender_targets)
        master_acc = self.acc(master_predict, master_targets)
        sub_acc = self.acc(sub_predict, sub_targets)
        color_acc = self.acc(color_predict, color_targets)

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

        print("gender_confusion_metric: ", gender_confusion_metric)
        print("master_confusion_metric: ", master_confusion_metric)
        print("sub_confusion_metric: ", sub_confusion_metric)
        print("color_confusion_metric: ", color_confusion_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['hparams']['lr'])
        return optimizer


if __name__ == '__main__':
    pass


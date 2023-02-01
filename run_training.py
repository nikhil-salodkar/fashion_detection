import os
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from dataset import FashionClassificationDataModule
from training import FashionPrediction


def get_data(path):
    """Read the pre-processed/cleaned dataset"""
    full_data = pd.read_csv(os.path.join(path, 'final-styles_df.csv'))
    return full_data


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-total-loss:.3f}-{val-total-acc:.3f}-{total-f1score:.3f}'
                                          , save_top_k=2, monitor='total-f1score', mode='max', save_last=False)
    wandb_logger = WandbLogger(project="fashion_classification", save_dir='lightning_logs', name='just_testing_run')
    early_stopping = EarlyStopping(monitor="total-f1score", patience=5, verbose=False, mode="max")
    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=50,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


def test_training():
    pl.seed_everything(15798)
    master_weights = [0.98306711, 0.51680662, 1.19520994, 4.736135]
    subcat_weights = [11.28714139, 14.05133929, 0.45237557, 1.69794236, 0.51593528,
                      12.99086085, 2.88081851, 32.02398256, 1.28334692, 1.5082489,
                      1.37154507, 4.86583481, 0.76544261, 1.27621061, 2.63294694,
                      2.98058712, 4.54465759, 36.23766447, 4.18550532, 1.42993899,
                      3.22489754, 11.66975636, 59.87092391, 0.18758088, 21.85763889,
                      22.57428279, 1.97282414, 15.30034722, 5.33733043, 0.08958632,
                      1.47591774, 0.54171174]
    gender_weights = [10.77383863, 13.70606532, 0.39973693, 4.2086915, 0.47733304]
    color_weights = [1.3442648, 0.10384459, 0.20509467, 10.65401354, 0.2888599,
                     23.84469697, 4.43131537, 32.30571848, 11.78208556, 2.63546651,
                     1.61268482, 0.47689394, 0.36603701, 6.85943337, 7.20487247,
                     6.18195847, 7.82404119, 1.7416996, 34.53369906, 23.84469697,
                     2.55478896, 62.59232955, 10.43205492, 0.56136618, 43.54249012,
                     5.50262238, 2.46063212, 1.90757576, 5.13578089, 0.54310047,
                     0.61515803, 0.4112843, 37.09175084, 15.17389807, 45.52169421,
                     0.92132224, 5.69021178, 3.18941807, 8.78488836, 91.04338843,
                     8.34564394, 14.72760695, 0.18238523, 1.305707]
    hparams = {'path': 'data/fashion-dataset/images/', 'batch_size': 64, 'gender_classes': 5, 'mastercat_classes': 4,
               'subcat_classes': 32, 'color_classes': 44, 'resnet_type': 'resnet152', 'layer1': 2048, 'layer2': 1024,
               'layer3': 256, 'activation': 'ReLU', 'dropout_val': 0.3, 'lr': 1e-4}

    gender_dict = {'Boys': 0, 'Girls': 1, 'Men': 2, 'Unisex': 3, 'Women': 4}
    master_dict = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}
    sub_dict = {'Accessories': 0, 'Apparel Set': 1, 'Bags': 2, 'Belts': 3, 'Bottomwear': 4, 'Cufflinks': 5, 'Dress': 6,
                'Eyes': 7, 'Eyewear': 8, 'Flip Flops': 9, 'Fragrance': 10, 'Headwear': 11, 'Innerwear': 12,
                'Jewellery': 13, 'Lips': 14,
                'Loungewear and Nightwear': 15, 'Makeup': 16, 'Mufflers': 17, 'Nails': 18, 'Sandal': 19, 'Saree': 20,
                'Scarves': 21,
                'Shoe Accessories': 22, 'Shoes': 23, 'Skin': 24, 'Skin Care': 25, 'Socks': 26, 'Stoles': 27, 'Ties': 28,
                'Topwear': 29,
                'Wallets': 30, 'Watches': 31}
    color_dict = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Burgundy': 5, 'Charcoal': 6,
                  'Coffee Brown': 7,
                  'Copper': 8, 'Cream': 9, 'Gold': 10, 'Green': 11, 'Grey': 12, 'Grey Melange': 13, 'Khaki': 14,
                  'Lavender': 15,
                  'Magenta': 16, 'Maroon': 17, 'Mauve': 18, 'Metallic': 19, 'Multi': 20, 'Mushroom Brown': 21,
                  'Mustard': 22, 'Navy Blue': 23,
                  'Nude': 24, 'Off White': 25, 'Olive': 26, 'Orange': 27, 'Peach': 28, 'Pink': 29, 'Purple': 30,
                  'Red': 31, 'Rose': 32,
                  'Rust': 33, 'Sea Green': 34, 'Silver': 35, 'Skin': 36, 'Steel': 37, 'Tan': 38, 'Taupe': 39,
                  'Teal': 40, 'Turquoise Blue': 41,
                  'White': 42, 'Yellow': 43}

    full_df = get_data(os.path.join('data/fashion-dataset/'))
    data_module = FashionClassificationDataModule(hparams, gender_dict, master_dict, sub_dict, color_dict, full_df)

    train_module = FashionPrediction(hparams, master_weights, subcat_weights, color_weights, gender_weights)
    train_model(train_module, data_module)


if __name__ == '__main__':
    test_training()

import os
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from dataset import FashionClassificationDataModule
from training import FashionPrediction


def get_data(path):
    full_data = pd.read_csv(os.path.join(path, 'final-styles_df.csv'))
    return full_data


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-total-loss:.3f}-{val-total-acc:.3f}-{total-f1score:.3f}'
                                          , save_top_k=2, monitor='total-f1score', mode='max', save_last=False)
    wandb_logger = WandbLogger(project="fashion_classification", save_dir='lightning_logs', name='first_run_64batch')
    early_stopping = EarlyStopping(monitor="total-f1score", patience=5, verbose=False, mode="max")
    # trainer = pl.Trainer(accelerator='gpu', fast_dev_run=2, max_epochs=5,
    #                      callbacks=[checkpoint_callback, early_stopping], precision=16)
    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=50,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


def test_training():
    pl.seed_everything(15798)
    hparams = {'path': 'data/fashion-dataset/images/', 'batch_size': 64, 'gender_classes': 5, 'mastercat_classes': 4,
               'subcat_classes': 32, 'color_classes': 44, 'resnet_type': 'resnet101', 'layer1': 2048, 'layer2': 1024,
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

    train_module = FashionPrediction(hparams)
    train_model(train_module, data_module)


if __name__ == '__main__':
    test_training()

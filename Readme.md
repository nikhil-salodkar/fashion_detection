## Problem Statement
Create and train a classification model using Convolutional Nerual Network architectures like
ResNet to classify images into different product categories and subcategories and use
the trained Neural Network to generate images embeddings which could help in creating a 
simple visual similarity recommendation system.
## Dataset and Preprocessing
The training Dataset used is taken from Kaggle. 

Dataset Link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

**Dataset summary**: 
This dataset has multiple labels which could be classified. The ones which are being
classified here are given below:
- Master Category:
  - Number of classes: 4
  - Labels of classes: 'Accessories', 'Apparel', 'Footwear', 'Personal Care'
- Sub Category:
  - Number of classes: 32
  - Labels of classes: 'Accessories', 'Apparel Set', 'Bags', 'Belts', 'Bottomwear', 'Cufflinks', 'Dress', 'Eyes', 'Eyewear', 'Flip Flops', 'Fragrance', 'Headwear', 'Innerwear', 'Jewellery', 'Lips', 'Loungewear and Nightwear', 'Makeup', 'Mufflers', 'Nails', 'Sandal', 'Saree', 'Scarves', 'Shoe Accessories', 'Shoes', 'Skin', 'Skin Care', 'Socks', 'Stoles', 'Ties', 'Topwear', 'Wallets', 'Watches'
- Product color:
  - Number of classes: 44
  - Labels: 'Beige', 'Black', 'Blue', 'Bronze', 'Brown', 'Burgundy', 'Charcoal', 'Coffee Brown', 'Copper', 'Cream', 'Gold', 'Green', 'Grey', 'Grey Melange', 'Khaki', 'Lavender', 'Magenta', 'Maroon', 'Mauve', 'Metallic', 'Multi', 'Mushroom Brown', 'Mustard', 'Navy Blue', 'Nude', 'Off White', 'Olive', 'Orange', 'Peach', 'Pink', 'Purple', 'Red', 'Rose', 'Rust', 'Sea Green', 'Silver', 'Skin', 'Steel', 'Tan', 'Taupe', 'Teal', 'Turquoise Blue', 'White', 'Yellow'
  - Gender/target person type:
    - Number of classes: 5
    - Labels: 'Boys', 'Girls', 'Men', 'Unisex', 'Women'

As can be seen from the dataset, the sheer number of classes in subcategories and color
can make accurately predicting correct class challenging.

There are other attributes to products including article type, season, year, and product
display name which can be used for more experiments but not used in current work.

The hierarchy of subcategories and article type may be not strict and further processing
of data will be required if hierarchical classification is to be designed.

Some data cleaning and processing can be found in the jupyter notebook.
## Modelling
Information about model architecture used can be found in modelling.py

**Fine-tuning:**
A pre-trained ResNet model is fine-tuned with linear layers corresponding to each
set of classification categories. ResNet architecture is kept in separate class from
linear layers so that after training image embeddings from convolution layers can be
extracted.

**Visual Similarity search:** After training of classification model is done, embeddings
extracted from ResNet are used to generate an index of existing product embeddings, which
can be searched for visual similarity given a new image embeddings as input.

The code to generate indexes using Faiss library can be found in the huggingface
demo code.

## Evaluation
Evaluation metrics can be visualized in weights and biases link [here](https://wandb.ai/nikhilsalodkar/fashion_classification?workspace=user-nikhilsalodkar).
As can be observed the model is especially struggling to get high F1 score for color
attribute. All other atrributes are getting quite good metric. This may be because there
are very large number of colors classes and there may be lots of color imbalance
## Demo
A demo link build using streamlit and deployed on huggingface is [here](https://huggingface.co/spaces/niks-salodkar/Fashion-Prediction-Demo).
You might have to restart the huggingface space and wait a short while to try the demo.

## Requirements
The required packages can be viewed in reqs.txt. This file could include extra packages
which might not be necessary. A new conda environment is recommended if you want to
test out on your own.
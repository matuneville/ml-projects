# ml-projects

This repository contains a variety of Machine Learning projects and implementations, including both complete and simple classification and prediction tasks, as well as foundational algorithms developed from scratch. It also features comprehensive analyses and typical data science workflows, and showcases practical applications and theoretical insights across different machine learning methodologies.

## Cool projects


- [**U-Net Cloud Segmentation**](/deep-learning/satellite-images-cloud-segmentation): This project implements a U-Net model from scratch for segmenting clouds in satellite images, based on the architecture from the [original U-Net paper for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597). The model is designed to handle multi-channel inputs (RGB and near-infrared) for accurate cloud classification, providing pixel-wise segmentation of clouds in satellite imagery. _The implementation is complete, but the model is pending training and fine-tuning due to the high computational cost involved_.


- **IMDb Review Classification**: This task involves categorizing IMDb movie reviews as positive or negative using various Natural Language Processing techniques.
  - [**RNN with LSTM**](deep-learning/imdb-sentiment-predict/RNN-classifier.ipynb): This approach utilizes recurrent neural networks (RNNs) with long short-term memory (LSTM) cells to effectively capture the sequential nature of text data, improving classification with an accuracy of 0.87.
  - [**Traditional and Out-of-Core Learning (Logistic approach)**](supervised-learning/imdb-review-classification/imdb.ipynb): This section examines traditional machine learning techniques for review classification, along with out-of-core learning methods designed for handling larger datasets with more efficiency. Additionally, it explores a potential unsupervised topic classification method for further categorization of reviews.


- [**CNN-based Smile Recognition**](/deep-learning/CelebA-attributes-classification/CelebA-attributes-classification.ipynb): A Convolutional Neural Network model trained on the CelebA dataset. This model focuses on classifying whether a face in a given image is smiling or not. The project demonstrates data preprocessing, model training, and evaluation, providing insights into deep learning techniques for image classification tasks.


- [**Insurance Market Customer Segmentation**](unsupervised-learning/insurance-market-segmentation/insurance-market-segmentation.ipynb): An unsupervised learning project focused on segmenting customers in the insurance market. It also includes an efficient supervised classifier based on the cluster predictions.


- [**CABA Apartments Price Prediction**](supervised-learning/caba-apartment-price-prediction/caba-apartments.ipynb): A predictive modeling project focused on forecasting apartment prices in Buenos Aires, employing various regression techniques.

### Main sources

- _**Machine Learning with PyTorch and Scikit-Learn**_. Book by Liu Yuxi, Sebastian Raschka, and Vahid Mirjalili
- _**Practical Statistics for Data Scientists**: 50+ Essential Concepts Using R and Python_. Book by Andrew Bruce, Peter Bruce, and Peter Gedeck


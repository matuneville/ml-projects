# ml-projects

This repository contains a variety of Machine Learning projects and implementations, including both complete and simple classification and prediction tasks, as well as foundational algorithms developed from scratch. It also features comprehensive analyses and typical data science workflows, and showcases practical applications and theoretical insights across different machine learning methodologies.

## Some projects

### **Natural Language Processing (NLP)**:
  - [**Sentence Grammar Acceptability (Español)**](NLP-with-transformers/BERT_spanish_sentence_acceptability/fine-tuning_BERT.ipynb): This project fine-tunes a BERT model to evaluate the grammatical acceptability of sentences in Spanish. Starting with a text pre-processing, that includes tokenizing sentences with a pre-trained BERT tokenizer, padding tokens, and creating attention masks, the model is then trained on labeled data. The model learns to classify sentences with a high level of accuracy.

  - [**Character-Level Language Modeling**](/deep-learning/character-level-language-modelling/character-prediction-with-book.ipynb): This project uses an LSTM-based RNN to generate text at the character level, trained on H.P. Lovecraft's "At the Mountains of Madness." The model learns to predict each next character, capturing word patterns and often producing coherent sequences. The bidirectional LSTM enhances context by processing input in both directions, improving text quality. In the final evaluation, the model shows interesting results, generating grammatically correct expressions and occasionally surprising coherent text outputs.

  - **IMDb Review Classification**: This task involves categorizing IMDb movie reviews as positive or negative using various Natural Language Processing techniques.
    - [**BERT Fine-Tuned Model**](deep-learning/imdb_analysis_BERT/imdb_analysis_BERT.ipynb): This fine-tuning approach leverages [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) (simpler BERT model from Hugging Face) to classify IMDb reviews with, reaching an accuracy of 0.92. While the model could achieve even higher accuracy on the full dataset, training efficiency was prioritized due to the computational expense of processing the entire dataset.
    - [**RNN with LSTM**](deep-learning/imdb_analysis_RNN.ipynb): This approach utilizes recurrent neural networks (RNNs) with long short-term memory (LSTM) cells to effectively capture the sequential nature of text data, improving classification with an accuracy of 0.87.
    - [**Traditional and Out-of-Core Learning (Logistic approach)**](supervised-learning/imdb-review-classification/imdb.ipynb): This approach implements traditional machine learning techniques used in NLP for review classification, along with out-of-core learning methods designed for handling larger datasets more efficiently (accuracies of 0.90 and 0.87 respectively). Additionally, it explores a potential unsupervised topic classification method for further categorization of reviews.

### **Computer Vision**
  - [**U-Net Cloud Segmentation**](/deep-learning/satellite-images-cloud-segmentation): This project implements a U-Net model from scratch for segmenting clouds in satellite images, based on the architecture from the [original U-Net paper for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597). The model is designed to handle multi-channel inputs (RGB and near-infrared) for accurate cloud classification, providing pixel-wise segmentation of clouds in satellite imagery. _The implementation is complete, but the model is pending training and fine-tuning due to the high computational cost involved_.

  - [**CNN-based Smile Recognition**](/deep-learning/CelebA-attributes-classification/CelebA-attributes-classification.ipynb): A Convolutional Neural Network model trained on the CelebA dataset. This model focuses on classifying whether a face in a given image is smiling or not. The project demonstrates data preprocessing, model training, and evaluation, providing insights into deep learning techniques for image classification tasks.

### **Unsupervised Learning**:
  - [**Insurance Market Customer Segmentation**](unsupervised-learning/insurance-market-segmentation/insurance-market-segmentation.ipynb): An unsupervised learning project focused on segmenting customers in the insurance market. It also includes an efficient supervised classifier based on the cluster predictions.

### **Supervised Learning**:
  - [**CABA Apartments Price Prediction**](supervised-learning/caba-apartment-price-prediction/caba-apartments.ipynb): A predictive modeling project focused on forecasting apartment prices in Buenos Aires, employing various regression techniques.
  - [**Heart Attack Prediction**](supervised-learning/heart-attack-prediction-&-analysis/heart-attack.ipynb): This project predicts the likelihood of heart disease based on clinical data, emphasizing models and techniques to maximize recall. It includes data preprocessing, feature visualization, and performance evaluation to ensure high sensitivity in detecting potential cases.

### Main sources

- _**Machine Learning with PyTorch and Scikit-Learn**_. Book by Liu Yuxi, Sebastian Raschka, and Vahid Mirjalili
- _**Transformers for Natural Language Processing and Computer Vision - Third Edition**_. Denis Rothman
- _**Practical Statistics for Data Scientists**: 50+ Essential Concepts Using R and Python_. Book by Andrew Bruce, Peter Bruce, and Peter Gedeck

Additional papers and Medium articles are referenced within the specific projects when applied.
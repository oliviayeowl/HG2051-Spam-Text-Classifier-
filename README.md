
# Spam Classification with Cross-Domain Evaluation

## Overview

This project implements a **Naive Bayes spam classifier** designed to evaluate the robustness of spam detection across different text domains. The model is trained on **SMS spam data** and evaluated on **email data**, allowing us to examine how well a classifier trained in one domain generalises to another.

In addition to traditional bag-of-words approaches, the classifier incorporates a **custom set of linguistically motivated features** extracted from each message. The project therefore explores both **feature engineering and cross-domain transferability** in spam classification.

## Project Objective

The primary goal of this project is to investigate:

• how well a spam classifier trained on SMS messages performs when applied to email data
• whether linguistically motivated features improve classification performance
• the robustness of the model across repeated training and evaluation cycles

## Main Script

### `spam_classifier.py`

This is the primary script responsible for loading datasets, preprocessing text, extracting features, training the classifier, and evaluating performance.

The script performs the following steps:

### 1. Data Loading

The script loads two datasets:

• **SMS Spam Dataset**
`datasets/SMSSpamCollection.txt`

• **Email Corpus**
`datasets/email_corpus_lingspam.txt`

The SMS dataset is used for training and testing, while the email dataset is used to evaluate cross-domain generalisation.

### 2. Preprocessing

Text data undergoes several preprocessing steps:

• tokenisation of messages
• removal of stopwords
• removal of names and nouns
• construction of the top **100 most frequent ham and spam words**

These preprocessing steps are designed to reduce noise and focus on features that are more predictive of spam content.

### 3. Feature Engineering

For each message, the function:

```
extract_features(text)
```

extracts **five linguistically motivated features**.

These features are then applied to both the SMS and email datasets to generate the input vectors used for classification.

### 4. Model Training

The model uses a **Naive Bayes classifier**.

Training procedure:

• 80% of the SMS dataset is used for training
• 20% is reserved as a test set

### 5. Evaluation

Model performance is evaluated on two datasets:

• the **SMS test set** (in-domain evaluation)
• the **full email dataset** (cross-domain evaluation)

This allows comparison between **within-domain performance and cross-domain transferability**.

### 6. Robustness Testing

To assess model stability, the training and evaluation process is repeated **10 times**.

For each iteration:

• the classifier is retrained
• accuracy is evaluated on the email dataset
• results are logged to `output.txt`

This helps evaluate how consistent the model's cross-domain performance is across multiple runs.

## Repository Structure

```
project/
│
├── spam_classifier.py
├── datasets/
│   ├── SMSSpamCollection.txt
│   └── email_corpus_lingspam.txt
│
├── output.txt
└── README.md
```

## Output

The file:

```
output.txt
```

stores the classification accuracy from each of the **10 evaluation runs** on the email dataset.

This provides a simple measure of the classifier’s **cross-domain robustness**.

## Key Concepts

This project touches on several core NLP concepts:

• spam detection
• Naive Bayes classification
• feature engineering for text classification
• cross-domain generalisation
• robustness testing in machine learning
# Abstract Classification with BERT

This is my first machine learning project (although i published to github really late), where I built an abstract classification system using NLP techniques. The goal is to classify scientific paper abstracts into one of three fields: **Physics**, **Mathematics**, or **Cybersecurity**. I implemented and fine-tuned a BERT model and compared its performance against traditional machine learning classifiers.

## Project Overview

Manual classification of scientific abstracts is time-consuming and inconsistent. This project explores the use of BERT (Bidirectional Encoder Representations from Transformers) to automatically classify paper abstracts into three academic domains.

## Methodology

- **Dataset**: 1500 abstracts (500 per category)
- **Preprocessing**: Tokenization, padding, and attention masks
- **Model**: Fine-tuned BERT with a classification head
- **Validation**: 5-Fold Stratified Cross-Validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix

## Experimental Setup

I experimented with:
- Classical classifiers: Logistic Regression, Random Forest, Decision Tree
- BERT with varying:
  - Epochs (5, 10, 30)
  - Batch sizes (4, 8, 16, 32)

## Results Summary

| Model               | Accuracy (%) |
|---------------------|--------------|
| Decision Tree       | 70.22        |
| Random Forest       | 89.22        |
| Logistic Regression | 90.11        |
| **BERT**            | **97.00**    |

- Optimal BERT config: **Epoch = 5**, **Batch size = 16**
- Best F1-score: **99.33**
- BERT significantly outperformed traditional models, especially in complex text understanding.

---

*Note: As this is my first ML project, there may be some technical limitations. I appreciate your understanding.*
*For more details, please refer to the full report. [PDF included in this repository](https://github.com/Manami108/abstract-classification/blob/main/results.pdf)*


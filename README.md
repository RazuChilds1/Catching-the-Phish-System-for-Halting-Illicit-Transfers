# Fraud Detection and Classification in Consumer Financial Complaints

## Overview

This project builds a **two-stage machine learning system** for identifying fraud-related complaints in the **CFPB Consumer Complaint Database**.

Rather than assuming the complaint data is already a clean fraud dataset, this project treats it as what it really is:

> a large, noisy, real-world complaint dataset where fraud-related cases are mixed with reporting disputes, account issues, servicing problems, and other non-fraud complaints.

The system is designed to solve this in two stages:

1. **Stage 1 — Fraud Detection**  
   Predict whether a complaint is **fraud-related** or **not fraud-related**

2. **Stage 2 — Fraud Type Classification**  
   For complaints identified as fraud-related, classify the complaint into a more specific fraud category

---

## Problem Statement

Consumer complaint data is high-volume, text-heavy, and noisy. Fraud-related complaints are often buried inside broader complaint categories, making manual review difficult and inefficient.

This project aims to build a machine learning pipeline that can:

- detect fraud-related complaints from complaint narratives
- separate fraud signals from general complaint traffic
- support future risk analysis, complaint triage, or investigation workflows

---

## Dataset

Source: **CFPB Consumer Complaint Database**

The dataset contains consumer-submitted complaints about financial products and services.

### Key fields used
- `Consumer complaint narrative`
- `Product`
- `State`
- `Issue`
- `Sub-issue`
- `Date received`

### Data characteristics
- large-scale real-world complaint data
- contains both structured and unstructured features
- highly noisy and partially ambiguous
- includes both fraud-related and non-fraud complaints

---

## Project Framing

A major challenge in this dataset is that complaint issues that appear fraud-related are not always pure fraud cases.

For example, the data may include:
- fraud or scam complaints
- identity theft complaints
- unauthorized transaction complaints
- credit reporting disputes
- account management complaints
- payment process issues
- general servicing complaints

Because of this, the project is framed as a **fraud detection problem inside noisy complaint data**, not just a simple complaint classifier.

---

## System Design

### Stage 1 — Fraud Detection
Binary classification:

- `1` = fraud-related
- `0` = not fraud-related

This stage focuses on identifying whether a complaint likely reflects malicious or deceptive activity.

Examples of fraud-related signals:
- fraud
- scam
- identity theft
- unauthorized transactions
- debt tied to identity theft
- fraud alerts / security freeze complaints

Examples of non-fraud signals:
- incorrect reporting
- improper report use
- account management issues
- payment process issues
- general dispute complaints

### Stage 2 — Fraud Type Classification
Multi-class classification for complaints already identified as fraud-related.

Potential fraud categories include:
- fraud
- identity_theft
- scam
- transaction_issue
- debt_fraud
- security_issue

---

## Current Progress

### Completed
- loaded the full CFPB complaint dataset
- handled initial file/parsing issues
- selected relevant columns
- cleaned complaint narratives
- removed missing and duplicate complaint text
- performed exploratory data analysis (EDA)
- built an initial labeling strategy
- redefined the project as a two-stage system
- created a **Stage 1 high-confidence binary fraud dataset**

### Stage 1 dataset summary
The current Stage 1 dataset contains:

- **634,878 total complaints**
- **529,039 non-fraud complaints**
- **105,839 fraud-related complaints**

Approximate class distribution:
- **83.3% non-fraud**
- **16.7% fraud**

This is a realistic and manageable imbalance for baseline modeling.

---

## Exploratory Data Analysis

EDA focused on understanding:

- class distribution
- complaint volume over time
- narrative length
- dominant products and issues
- label imbalance
- noise and ambiguity in complaint categories

Key findings:
- the original complaint data is highly noisy
- some complaint categories overlap conceptually
- fraud-related complaints are a minority of the broader dataset
- complaint narratives are long enough to support NLP modeling
- label ambiguity makes direct multi-class classification difficult without a staged approach

---

## Labeling Strategy

For Stage 1, the dataset was converted into a **binary classification problem**.

### High-confidence fraud signals
Examples:
- fraud
- scam
- identity theft
- unauthorized transactions
- debt resulting from identity theft
- security freeze / fraud alert issues

### High-confidence non-fraud signals
Examples:
- incorrect information on your report
- improper use of your report
- managing an account
- closing an account
- trouble during payment process
- purchase dispute issues

### Important note
Ambiguous rows were removed from the Stage 1 training set in order to create a cleaner supervised learning problem.

This makes Stage 1 a **high-confidence training dataset**, not a perfect representation of all complaints in production.

---

## Modeling Plan

### Stage 1 — Fraud Detection
Initial baseline models:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM

Planned advanced models:
- DistilBERT
- BERT

Potential structured features:
- Product
- State
- Date-derived features
- narrative length

### Stage 2 — Fraud Type Classification
Models will be trained only on fraud-related complaints.

Possible approaches:
- TF-IDF + Logistic Regression
- transformer-based NLP models
- hybrid models combining text + structured features

---

## Evaluation Strategy

Because the classes are imbalanced, **accuracy is not the main metric**.

Primary evaluation metrics:
- Precision
- Recall
- F1-score

Why:
- a model that predicts “not fraud” too often may still achieve high accuracy
- fraud detection requires careful balance between catching fraud and limiting false positives

---

## Leakage Considerations

The project takes care to avoid feature leakage.

Fields used to derive labels, especially:
- `Issue`
- `Sub-issue`

should **not** be used directly as model inputs for Stage 1, because that would allow the model to learn the labeling rules rather than the complaint patterns.

Safer Stage 1 inputs:
- complaint narrative text
- product
- state
- date-derived features

---

## Repository Structure

Example structure:

```bash
project/
│
├── data/
│   ├── raw/
│   │   └── complaints.csv
│   ├── processed/
│   │   ├── cfpb_clean.parquet
│   │   ├── cfpb_eda_ready.parquet
│   │   └── cfpb_stage1.parquet
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_stage1_dataset_build.ipynb
│   └── 04_stage1_modeling.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── metrics/
│
├── README.md
└── requirements.txt

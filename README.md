# RNN_Transformer_Attention

# CS5760 Natural Language Processing – Homework 4

Student Name: Rohith Varma Suraparaju  
Course: CS5760 Natural Language Processing  
University: University of Central Missouri  
Semester: Spring 2026  

---

# Overview

This project implements several neural network architectures used in Natural Language Processing:

- Character-Level Recurrent Neural Network (RNN)
- Mini Transformer Encoder
- Scaled Dot-Product Attention

All implementations are provided using **Python and Jupyter Notebook**.



# Q1 – Character-Level RNN Language Model

## Goal

Train a character-level RNN to predict the next character given previous characters.

## Dataset

A toy dataset was created containing simple words and phrases such as:

```
hello
help
hello world
hello rnn
```


## Observations

- Lower temperature produces more predictable text.
- Higher temperature produces more diverse but noisy outputs.
- Larger hidden sizes improve representation but increase training time.

---

# Q2 – Mini Transformer Encoder

## Goal

Implement a simplified Transformer encoder to process sentences using self-attention.

## Dataset

Example sentences used:

```
machine learning is fun
deep learning uses neural networks
transformers use attention
nlp models process language
```

## Steps Implemented

```
1. Tokenize sentences
2. Convert words into numerical tokens
3. Apply embeddings
4. Implement self-attention
5. Demonstrate attention weights
```
---

# Q3 – Scaled Dot-Product Attention

## Goal

Implement the attention mechanism used in Transformer models.

## Attention Formula

```
Attention(Q, K, V) = softmax((QKᵀ) / √d_k) V
```

Where:

```
Q = Query matrix
K = Key matrix
V = Value matrix
d_k = dimension of key vectors
```

## Implementation Steps

```
1. Generate random Q, K, V matrices
2. Compute dot-product scores
3. Apply scaling factor √d_k
4. Apply softmax
5. Multiply attention weights with V
```

Scaling prevents large values that could destabilize softmax.

# Conclusion

This assignment demonstrates the core components of modern NLP systems including sequence models and attention mechanisms. The RNN model captures sequential patterns in text while the Transformer attention mechanism learns relationships between tokens.

These experiments highlight the importance of embeddings, sequence modeling, and attention mechanisms in natural language processing.

---

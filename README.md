Task 1: News Topic Classifier Using BERT

Objective of the Task

The goal was to build a robust Natural Language Processing (NLP) model to classify news headlines into four distinct categories: World, Sports, Business, and Sci/Tech. This task focused on leveraging state-of-the-art Transformer models for text classification.

Methodology / Approach
Dataset: Utilized the AG News Dataset from Hugging Face.
Preprocessing: Implemented tokenization using BertTokenizer, including padding and truncation to handle varying headline lengths.
Architecture: Fine-tuned the bert-base-uncased model using the Hugging Face Trainer API.
Optimization: Employed Transfer Learning to adapt the pre-trained BERT weights to the specific news classification task.
Deployment: Created a live interaction interface using Gradio to allow users to input headlines and see real-time predictions.
Key Results or Observations
Performance: The model achieved high accuracy and a strong F1-score, significantly outperforming traditional machine learning models (like Naive Bayes).
Observation: Fine-tuning BERT proved that "bidirectional context" is crucial for understanding short headlines where keyword-based approaches often fail.

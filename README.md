📰 Task 1: News Topic Classification Using BERT
🎯 Objective

To build a robust Natural Language Processing (NLP) model capable of classifying news headlines into four categories:

World

Sports

Business

Sci/Tech

The task emphasizes the use of state-of-the-art Transformer models for text classification.

🛠 Methodology / Approach

Dataset: AG News Dataset (Hugging Face)

Preprocessing:

Tokenization using BertTokenizer

Applied padding and truncation to manage variable headline lengths

Model Architecture:

Fine-tuned bert-base-uncased using the Hugging Face Trainer API

Optimization:

Leveraged Transfer Learning to adapt pre-trained BERT weights for news classification

Deployment:

Built a live prediction interface using Gradio for real-time headline classification

📊 Key Results & Observations

Performance: Achieved high accuracy and strong F1-score, outperforming traditional ML models such as Naive Bayes

Insight: Bidirectional context in BERT is crucial for understanding short headlines where keyword-based models fail

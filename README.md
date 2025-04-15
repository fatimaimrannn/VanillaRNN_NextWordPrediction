Vanilla RNN for Next-Word Prediction
Objective
The objective of this project is to implement a Vanilla Recurrent Neural Network (RNN) for next-word prediction using a Shakespeare text dataset. Instead of relying on pre-trained word embeddings like Word2Vec or GloVe, the model will train its own word embeddings using an Embedding Layer in TensorFlow or PyTorch.

Dataset
We will be using the publicly available Shakespeare Text Dataset from Hugging Face. The dataset consists of Shakespeare's works, which can be used to predict the next word in a sequence.

Dataset: Shakespeare's Text Dataset

Source: Hugging Face - Shakespeare Dataset

Instructions
1. Load and Preprocess the Dataset
Load the dataset from Hugging Face.

Tokenize the words in the dataset and create a vocabulary.

Split the dataset into training (80%) and testing (20%).

2. Implement the Vanilla RNN Model
Custom RNN Cell: Implement a custom Vanilla RNN cell (no LSTMs or GRUs). You can use Python classes for this implementation.

Embedding Layer: Use a trainable Embedding Layer in TensorFlow or PyTorch to learn word representations. Set an appropriate embedding size.

Model Task: The RNN should process sequences of words and predict the next word in the sequence.

Loss Function: Use Cross-Entropy Loss to evaluate the model.

Optimizer: Use an appropriate optimizer such as Adam for training.

3. Train the Model and Monitor Performance
Training: Train the model using Backpropagation Through Time (BPTT).

Loss Monitoring: Track and monitor training loss and validation loss across epochs.

Model Saving: Save the trained model after completion of training.

4. Generate Text Predictions
Seed Phrase: Provide a seed phrase (e.g., "To be or not to").

Text Generation: The model should generate the next word iteratively.

Sentence Generation: Generate at least 10 words to form a complete sentence.

5. Evaluate Model Performance
Metrics: Compute and report the following evaluation metrics:

Perplexity: Measures the uncertainty of the model.

Word-level accuracy: Evaluate accuracy at the word level.

Loss Curve Visualization: Plot the loss curves during training and validation.

Embedding Comparison: Compare learned embeddings with randomly initialized embeddings.

6. Ablation Studies
Pretrained Embeddings: Train the model using pretrained embeddings (Word2Vec or GloVe) and compare its performance with the model using randomly initialized embeddings.

Metrics Comparison: Evaluate and compare:

Perplexity

Word-level accuracy

Loss curve visualization

Confusion Matrix: Plot a confusion matrix showing misclassified words.

Impact Analysis: Analyze and discuss how the use of pretrained embeddings impacts the model’s performance.

7. Expected Output
Table Comparison: A table comparing the performance of the model using learned embeddings vs. random embeddings:


Embedding Type	Word-Level Accuracy	Perplexity
Random Embeddings	[Value]	[Value]
Learned Embeddings	[Value]	[Value]
Generated Text: The model will generate text sequences based on a provided seed phrase.

Requirements
Python 3.x

TensorFlow or PyTorch

Matplotlib, Seaborn (for plotting)

NumPy, Pandas

Hugging Face Datasets (for loading the Shakespeare dataset)

How to Run the Code
Clone the repository.

Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Run the Python script or Jupyter notebook:

bash
Copy
Edit
python rnn_shakespeare.py
Notes
The project includes code for training a Vanilla RNN model on the Shakespeare dataset and generating next-word predictions based on seed phrases.

The comparison of learned vs. random embeddings helps evaluate the effectiveness of training embeddings directly from the data.


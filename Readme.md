
# Hate Speech Detection Project

## Project Overview

This project aims to develop a binary hate speech detection system based on social media text data. The process involves data preprocessing, dataset merging, and text cleaning, followed by building a machine learning model using LSTM neural networks.

## Key Steps

1. **Data Loading and Merging**:
   - Two datasets are loaded: `imbalanced_data.csv` and `raw_data.csv`. The datasets contain tweets labeled as hate speech, offensive language, or neither.
   - Unnecessary columns are removed, and the datasets are merged to create a combined dataset for training.

2. **Class Balancing**:
   - The classes are recoded to binary labels: hate speech/abusive (1) and non-hate speech (0).
   - Class distribution is analyzed to understand the label imbalance and plotted for visualization.

3. **Text Preprocessing**:
   - A comprehensive text cleaning function (`clean_tweet()`) is applied to remove emojis, special characters, stop words, and perform tokenization and stemming.
   - The cleaned text is prepared for further analysis by converting it into sequences using TensorFlow's `Tokenizer`.

4. **Model Architecture**:
   - The model is built using an LSTM neural network with an embedding layer and dropout for regularization.
   - A final dense layer with sigmoid activation is used for binary classification of the tweets.

5. **Training and Evaluation**:
   - The model is trained with early stopping and model checkpointing for efficient training management.
   - The model achieves a test accuracy of 94.7%, and performance metrics like accuracy and loss are visualized for both training and validation sets.

## Model Performance

- **Test Accuracy**: 94.7%
- **Training Epochs**: 5 (with early stopping)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

## Files

- `model.h5`: Saved model file.
- `tokenizer.pickle`: Saved tokenizer for text preprocessing.
- `best_model.keras`: Best performing model based on validation loss.

## How to Run

1. Install dependencies:
   ```
   pip install emoji nltk keras tensorflow
   ```

2. Download the necessary datasets and place them in the same directory as the script.

3. Run the model training script:
   ```
   python hate_speech_detection.py
   ```

4. Evaluate the model:
   ```
   python evaluate_model.py
   ```

## License

This project is licensed under the MIT License.

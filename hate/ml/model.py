# Creating model architecture.
from hate.entity.config_entity import ModelTrainerConfig
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding, SpatialDropout1D
from hate.constants import *

class ModelArchitecture:

    def __init__(self):
        pass

    def get_model(self):
        model = Sequential()
        # Ensure the input dimensions match the expected input data shape
        model.add(Embedding(input_dim=MAX_WORDS, output_dim=100))  # Removed input_length parameter
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation=ACTIVATION))  # Use sigmoid activation for binary classification
        
        # Print the model summary for verification
        model.summary()
        
        # Compile the model with appropriate loss function and optimizer
        model.compile(loss=LOSS, optimizer=RMSprop(), metrics=METRICS)

        return model

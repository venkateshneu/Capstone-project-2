import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


from keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts,DataTransformationArtifacts
from hate.ml.model import ModelArchitecture

class ModelTrainer:
    def __init__(self,data_transformation_artifacts: DataTransformationArtifacts,
                model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    
    def spliting_data(self, csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x and y")
            x = df[TWEET].astype(str)  # Ensure all entries in x are strings
            y = df[LABEL]

            logging.info("Applying train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            print(len(x_train), len(y_train))
            print(len(x_test), len(y_test))
            print(type(x_train), type(y_train))
            logging.info("Exited the spliting_data function")
            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) from e

    
    def tokenizing(self, x_train):
        try:
            logging.info("Applying tokenization on the data")
            x_train = x_train.dropna()  # Drop any NaN values
            x_train = x_train.apply(lambda x: str(x).lower())  # Ensure all data is in lower case string form

            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Converting text to sequences: {sequences[:5]}")  # Print the first 5 sequences for debugging
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"The sequence matrix shape is: {sequences_matrix.shape}")  # Check the shape of the sequences_matrix
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    

    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            logging.info("Entered the initiate_model_trainer function ")
            x_train, x_test, y_train, y_test = self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()   

            model = model_architecture.get_model()

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info(f"Training data shape (X_train): {sequences_matrix.shape}")
            logging.info(f"Labels shape (y_train): {y_train.shape}")

            logging.info("Entered into model training")
            # Add this logging statement before model.fit()
            logging.info(f"Shape of sequences_matrix: {sequences_matrix.shape}")

            model.fit(sequences_matrix, y_train, 
                    batch_size=self.model_trainer_config.BATCH_SIZE, 
                    epochs=self.model_trainer_config.EPOCH, 
                    validation_split=self.model_trainer_config.VALIDATION_SPLIT)
            logging.info("Model training finished")

            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            logging.info("Saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

from model import Model
from preprocessor import Preprocessor
import argparse
import pandas as pd
import json
import pickle


class Pipeline:
    """
        A class representing a pipeline for data preprocessing, model training, and predictions.

        Attributes:
            model (Model): The machine learning model.
            preprocessor (Preprocessor): The data preprocessor.
        Methods:
            run(X, test=False): Execute the pipeline to train the model or make predictions.
    """

    def __init__(self, ):
        """
            Initializes a new instance of the Pipeline class.
        """
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, X, test=False):
        """
            Execute the pipeline to train the model or make predictions.

            Args:
                X (pd.DataFrame): The input DataFrame.
                test (bool, optional): If True, run the pipeline in testing mode. Defaults to False.

            Returns:
                None
        """
        if test:  # Use the dataframe as testing dataset
            with open("trained_preprocessor.pkl", 'rb') as file:  # Utilize the trained preprocessor
                self.preprocessor = pickle.load(file)

            with open("trained_model.pkl", 'rb') as file:  # Utilize the trained model
                self.model = pickle.load(file)

            # According to information given to us during the class the test dataset itself should not
            # have the outcome related columns already, however if there is an issue with preprocessing
            # it is likely due to them not being removed
            # In that scenario we kindly ask you to remove the subsequent line fom comment
            # X = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'])

            X_test = self.preprocessor.transform(X)
            y_pred = self.model.predict_proba(X_test)

            # Save the outcome probabilities in a dictionary
            output = {
                'predict_probes': y_pred[:, 1].tolist(),
                'threshold': 0.5
            }

            with open('predictions.json', 'w') as output_file:
                json.dump(output, output_file, indent=2)

            print("Testing process finished")
            print("Predictions probabilities saved in 'prediction.json'")

        else:
            # Initialize the outcome column
            y_train = X['In-hospital_death']

            # Remove all outcome related columns from dataset
            X = X.drop(columns=['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'])

            # Train the preprocessor
            self.preprocessor.fit(X)
            with open("trained_preprocessor.pkl", 'wb') as file:  # Save the trained preprocessor for later use
                pickle.dump(self.preprocessor, file)

            print("Preprocessor training process finished")
            print("Preprocessor saved in 'trained_preprocessor.pkl'")

            X_train = self.preprocessor.transform(X)  # Utilize the preprocessor on test data
            self.model.fit(X_train, y_train)  # Train the model

            with open("trained_model.pkl", 'wb') as file:  # Save the trained model for later use
                pickle.dump(self.model, file)

            print("Model training process finished")
            print("Model saved in 'trained_model.pkl'")


if __name__ == "__main__":
    pipeline = Pipeline()  # Initialize the pipeline

    parser = argparse.ArgumentParser(description="This is the pipeline for our program")
    # Required type argument for data path, throws an error if not provided
    parser.add_argument("--data_path", type=str, required=True, help="Path for the input dataset")

    # Store True if test argument is given, otherwise: False
    parser.add_argument("--test", action="store_true", help="Run in testing mode.")

    args = parser.parse_args()  # Store the user inputted arguments

    df = pd.read_csv(args.data_path)  # Reads the dataframe from file
    pipeline.run(df, args.test)  # Activates the pipeline, using user arguments

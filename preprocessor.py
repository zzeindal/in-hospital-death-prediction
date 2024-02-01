from sklearn.experimental import enable_iterative_imputer
import pandas as pd
from sklearn.impute import IterativeImputer


class Preprocessor:
    """
        A class for preprocessing data using imputation and transformations.
    """

    def __init__(self, ):
        """
            Initializes a new instance of the Preprocessor class.
        """
        self.median_age = 0
        self.inputer = None
        pass

    def fit(self, df):
        """
            Fit the preprocessor on the provided DataFrame.
            Args:
                df (pd.DataFrame): The input DataFrame with missing values.
            Returns:
                pd.DataFrame: The processed DataFrame after imputation and transformations.
        """

        self.median_age = df['Age'].median()  # Saving the median age of training dataset
        # Fill the NaN values of gender according to their age and the training median age
        df['Gender'] = df['Gender'].fillna(value=(df['Age'] > self.median_age))
        df = df.drop(columns=['recordid'])  # Drop the column with no important information
        self.inputer = IterativeImputer()
        imputed_data = self.inputer.fit_transform(df)  # Fit the IterativeImputer for later filling the NaN values
        df_new = pd.DataFrame(imputed_data, columns=df.columns)  # Update the dataframe

    def transform(self, X):
        """
            Transform the provided DataFrame using the pre-fitted preprocessor.
            Args:
                X (pd.DataFrame): The input DataFrame with missing values.
            Returns:
                pd.DataFrame: The processed DataFrame after transformations.
        """
        # Fill the NaN values of gender according to their age and the training median age
        X['Gender'] = X['Gender'].fillna(value=(X['Age'] > self.median_age))
        X = X.drop(columns=['recordid'])  # Drop the column with no important information
        imputed_data = self.inputer.transform(X)  # Fill the NaN values in the rest of columns
        X = pd.DataFrame(imputed_data, columns=X.columns)  # Update the dataframe
        return X

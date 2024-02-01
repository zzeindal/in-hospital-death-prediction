from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class Model:
    """
        A class for creating and evaluating a Linear Discriminant Analysis model.

        Attributes:
            lda_model (LinearDiscriminantAnalysis): The Linear Discriminant Analysis model.

        Methods:
            fit(X_train, y_train): Fit the model on the training data.
            predict(X_test): Make predictions on new data.
            predict_proba(X_test): Get class probabilities for predictions.
            score(X_test, y_test): Evaluate the model's performance using F1 score, accuracy, recall, and precision.
    """

    def __init__(self):
        """
            Initializes a new instance of the Model class.
        """
        self.lda_model = LinearDiscriminantAnalysis()

    def fit(self, X_train, y_train):
        """
            Fit the model on the training data.
            Args:
                X_train (array-like or pd.DataFrame): The training data.
                y_train (array-like or pd.Series): The target values.

            Returns:
                None
        """
        self.lda_model.fit(X_train, y_train)

    def predict(self, X_test):
        """
            Make predictions on new data.

            Args:
                X_test (array-like or pd.DataFrame): The test data.

            Returns:
                array-like: Predicted class labels.
            """
        self.lda_preds = self.lda_model.predict(X_test)
        return self.lda_preds

    def predict_proba(self, X_test):
        """
            Get class probabilities for predictions.

            Args:
                X_test (array-like or pd.DataFrame): The test data.

            Returns:
                array-like: Class probabilities.
        """
        lda_preds = self.lda_model.predict_proba(X_test)
        return lda_preds

    def score(self, X_test, y_test):
        """
            Evaluate the model's performance using F1 score, accuracy, recall, and precision.

            Args:
                X_test (array-like or pd.DataFrame): The test data.
                y_test (array-like or pd.Series): The true labels.

            Returns:
                float: The F1 score of the model.
            """
        y_pred = self.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print("Model F1 Score:", f1_score(y_test, y_pred))
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print('Model Recall:', recall_score(y_test, y_pred))
        print("Model Precision:", precision_score(y_test, y_pred))
        return f1

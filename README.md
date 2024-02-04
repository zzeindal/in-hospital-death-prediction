# Project README

## Project Overview
This project involves the development of a machine learning pipeline for data preprocessing, model training, and testing. The pipeline consists of three main components: `preprocessor.py`, `model.py`, and `run_pipeline.py`. These files collectively create a structured workflow for handling and analyzing datasets.
It also contains files trained_model.pkl and trained_preprocessor.pkl containing pretrained preprocessor and model, which could be used for training proposes
Additionally we provided the dataset we trained the model on and the model predictions of the same dataset

## Group
Project provided was made by Team 1
Members:
- Aram Sargsian
- Karen Baghdasaryan
- Meri Gasparyan

### File Descriptions
1. **preprocessor.py**
   - Contains a class named `Preprocessor`.
   - The class includes `fit` and `transform` methods for handling preprocessing tasks such as handling missing values, scaling, feature extraction, etc.

2. **model.py**
   - Includes a class named `Model`.
   - The class contains `fit` and `predict` methods designed to work on preprocessed data.
   - Handles the training and prediction tasks for the machine learning model.

3. **pipeline.py**
   - Implements a class named `Pipeline`.
   - The `Pipeline` class includes a `run` method, which takes a data path (`--data_path`) and an optional testing mode argument (`--test`).
   - In testing mode, the outputs are saved in a `predictions.json` file, including predicted probabilities and a recommended threshold.
   - Handles saving and loading preprocessor and model instances for consistent testing and reproducing the final results.

### Usage Instructions
To run the pipeline, execute the following steps:

1. **Install Dependencies**
   - Install the necessary dependencies describes in requirements.txt.

2. **Run Pipeline**
   - Execute the pipeline by running `python run_pipeline.py --data_path <path_to_dataset> [--test]`.
   - Note that the program will trow an error in case the data path is not provided

### Additional Notes
- If running in testing mode, the pipeline will not fit the model. Ensure that the saved preprocessor and model instances are provided for testing purposes.
- In training mode, the data must include both features and labels. In testing mode, the data should not include labels.

# Sonar Rock vs Mine Prediction

## Overview
This project uses **Logistic Regression** to classify sonar signals as either **rock (R)** or **mine (M)** based on their frequency patterns. The dataset consists of **207 samples** with **60 numerical features** representing sonar readings.

## Dataset
- **Features:** 60 continuous values representing sonar signals.
- **Target Variable:**
  - `R` (Rock)
  - `M` (Mine)
- **Source:** Sonar signal classification dataset.

## Approach
1. **Data Preprocessing:**
   - Load the dataset from `Copy of sonar data.csv`.
   - No header, so manually assign column names if needed.
   - Convert categorical labels (`R`, `M`) into numerical format.

2. **Train-Test Split:**
   - Split the dataset into training (90%) and testing (10%) sets.
   - `X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)`

3. **Model Selection:**
   - Use **Logistic Regression**, a classification algorithm that predicts the probability of a sample belonging to a certain class.

4. **Model Training:**
   - Fit the model using the training data.
   - `model = LogisticRegression()`
   - `model.fit(X_train, Y_train)`

5. **Model Evaluation:**
   - Predict on test data.
   - Calculate accuracy using `accuracy_score()`.

## Dependencies
- `numpy`
- `pandas`
- `sklearn`

## Usage
Run the notebook `Rock vs Mine Prediction.ipynb` step by step to train and test the model.

## Results
The logistic regression model predicts whether an object is a **rock or a mine** based on sonar data. The accuracy score is displayed in the output of the notebook.

## Future Improvements
- Try different classification algorithms (SVM, Random Forest, Neural Networks).
- Tune hyperparameters for better performance.
- Feature engineering to improve model accuracy.


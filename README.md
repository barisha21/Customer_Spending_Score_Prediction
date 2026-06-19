# Customer Spending Score Prediction

## Project Overview

This Machine Learning project predicts a customer's Spending Score based on demographic and financial information such as age, income, profession, work experience, and family size.

The project follows a complete end-to-end Machine Learning workflow including data preprocessing, exploratory data analysis, feature engineering, model training, model comparison, evaluation, and deployment using Streamlit.

---

## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Feature Scaling
* Model Comparison
* Best Model Selection
* Model Evaluation
* Model Export and Reuse
* Streamlit Web Application

---

## Dataset Features

| Feature                | Description                |
| ---------------------- | -------------------------- |
| CustomerID             | Unique Customer Identifier |
| Gender                 | Male/Female                |
| Age                    | Customer Age               |
| Annual Income ($)      | Annual Income              |
| Profession             | Customer Profession        |
| Work Experience        | Years of Work Experience   |
| Family Size            | Number of Family Members   |
| Spending Score (1-100) | Target Variable            |

---

## Feature Engineering

Two additional features were created:

### Income Per Family

Income_Per_Family = Annual Income / Family Size

### Experience Ratio

Experience_Ratio = Work Experience / Age

These engineered features help improve model learning.

---

## Exploratory Data Analysis (EDA)

The project generates the following visualizations:

1. Age Distribution
2. Gender Count Plot
3. Income vs Spending Score
4. Family Size vs Income Box Plot
5. Pair Plot
6. Age vs Average Spending Score Trend
7. Correlation Heatmap
8. Gender Distribution Pie Chart
9. Average Work Experience by Gender
10. Spending Score Area Plot
11. Income by Family Size Violin Plot
12. Actual vs Predicted Plot

---

## Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Encoding Categorical Features
4. Feature Engineering
5. Train-Test Split
6. Feature Scaling
7. Model Training
8. Model Comparison
9. Best Model Selection
10. Model Evaluation
11. Model Saving
12. Deployment

---

## Models Compared

The following regression models were evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)

The model with the highest R² score is automatically selected and saved as the final model.

---

## Project Structure

Customer_Spending_Score_Prediction/

├── app.py

├── main.py

├── requirements.txt

├── README.md

├── data/

│ └── Customers.csv

├── models/

│ └── best_model.pkl

├── artifacts/

│ ├── scaler.pkl

│ ├── gender_encoder.pkl

│ └── profession_encoder.pkl

├── plots/

│ └── generated visualization files

└── src/

├── data_preprocessing.py

├── feature_engineering.py

├── train_model.py

├── predict.py

└── plots.py

---

## Installation

Install required libraries:

pip install -r requirements.txt

---

## Run the Project

### Train the Model

python main.py

### Launch Streamlit Application

python -m streamlit run app.py
---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Pickle

---

## Model Artifacts

The following files are generated after training:

* best_model.pkl
* scaler.pkl
* gender_encoder.pkl
* profession_encoder.pkl

These artifacts are reused during prediction and deployment.

---

## Future Improvements

* Hyperparameter Tuning using GridSearchCV
* Logging Support
* Advanced Model Evaluation
* Additional Feature Engineering
* Cloud Deployment

---

## Author

Machine Learning Project developed as part of Full Stack Internship and Machine Learning Practice.

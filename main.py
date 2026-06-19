from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.predict import predict_score
from src.plots import *

# Load data
df, gender_encoder, profession_encoder = preprocess_data(
    "data/Customers.csv"
)

# Feature Engineering
df = create_features(df)


# Train Model
model, scaler, y_test, y_pred = train_model(df, gender_encoder, profession_encoder)

# Evaluation Plot
actual_vs_predicted(y_test, y_pred)

# Plots
age_distribution(df)
gender_count(df)
income_vs_spending(df)
family_size_income(df)
pair_plot(df)
age_spending_trend(df)
correlation_heatmap(df)
gender_distribution(df)
work_experience_gender(df)
spending_score_area(df)
income_family_violin(df)


print("Project Completed Successfully")
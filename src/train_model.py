from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os


def train_model(df, gender_encoder, profession_encoder):

    # Remove CustomerID if present
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)

    # Features and Target
    X = df.drop('Spending Score (1-100)', axis=1)
    y = df['Spending Score (1-100)']

    print("Features Used:")
    print(X.columns.tolist())
    print("Number of Features:", len(X.columns))

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    best_model = None
    best_score = float('-inf')

    print("\nMODEL COMPARISON")
    print("-" * 50)

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(name)
        print("MAE:", mae)
        print("MSE:", mse)
        print("R2 Score:", r2)
        print("-" * 50)

        if r2 > best_score:
            best_score = r2
            best_model = model

    print(f"\nBest Model Selected: {type(best_model).__name__}")
    print(f"Best R2 Score: {best_score}")

    y_pred = best_model.predict(X_test_scaled)

    # Create folders
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)

    # Save best model
    with open('artifacts/models/best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    # Save scaler
    with open('artifacts/models/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Save encoders
    with open('artifacts/models/gender_encoder.pkl', 'wb') as file:
        pickle.dump(gender_encoder, file)

    with open('artifacts/models/profession_encoder.pkl', 'wb') as file:
        pickle.dump(profession_encoder, file)

    return best_model, scaler, y_test, y_pred
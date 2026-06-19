import pickle
import numpy as np

def predict_score(
    gender,
    age,
    income,
    profession,
    experience,
    family_size,
    income_per_family,
    experience_ratio
):

    with open('artifacts/models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('artifacts/models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    with open('artifacts/models/gender_encoder.pkl', 'rb') as file:
        gender_encoder = pickle.load(file)

    with open('artifacts/models/profession_encoder.pkl', 'rb') as file:
        profession_encoder = pickle.load(file)

    gender = gender_encoder.transform([gender])[0]
    profession = profession_encoder.transform([profession])[0]

    data = np.array([[
        gender,
        age,
        income,
        profession,
        experience,
        family_size,
        income_per_family,
        experience_ratio
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    return prediction[0]
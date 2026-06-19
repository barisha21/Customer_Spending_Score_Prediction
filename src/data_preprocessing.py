import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove null values
    df = df.dropna(subset=['Profession'])

    # Encoding
    gender_encoder = LabelEncoder()
    profession_encoder = LabelEncoder()

    df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    df['Profession'] = profession_encoder.fit_transform(df['Profession'])

    return df, gender_encoder, profession_encoder
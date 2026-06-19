def create_features(df):

    df['Income_Per_Family'] = (
        df['Annual Income ($)'] /
        df['Family Size'].replace(0, 1)
    )

    df['Experience_Ratio'] = (
        df['Work Experience'] /
        df['Age'].replace(0, 1)
    )

    return df
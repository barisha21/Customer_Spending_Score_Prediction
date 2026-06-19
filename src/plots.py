import matplotlib.pyplot as plt
import seaborn as sns


# 1. Age Distribution
def age_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=10, kde=True)
    plt.title('Age Distribution')
    plt.savefig('artifacts/plots/age_distribution.png')
    plt.close()


# 2. Gender Count
def gender_count(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Gender', data=df)
    plt.title('Gender Count')
    plt.savefig('artifacts/plots/gender_count.png')
    plt.close()


# 3. Income vs Spending Score
def income_vs_spending(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x='Annual Income ($)',
        y='Spending Score (1-100)',
        hue='Gender',
        data=df
    )
    plt.title('Income vs Spending Score')
    plt.savefig('artifacts/plots/income_vs_spending.png')
    plt.close()


# 4. Family Size vs Income Box Plot
def family_size_income(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x='Family Size',
        y='Annual Income ($)',
        data=df
    )
    plt.title('Income by Family Size')
    plt.savefig('artifacts/plots/family_size_income.png')
    plt.close()


# 5. Pair Plot
def pair_plot(df):
    sns.pairplot(
        df[
            [
                'Age',
                'Annual Income ($)',
                'Spending Score (1-100)',
                'Work Experience',
                'Family Size',
                'Gender'
            ]
        ],
        hue='Gender'
    )
    plt.savefig('artifacts/plots/pair_plot.png')
    plt.close()


# 6. Age vs Average Spending Score Trend
def age_spending_trend(df):
    avg_spend = df.groupby('Age')['Spending Score (1-100)'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        x='Age',
        y='Spending Score (1-100)',
        data=avg_spend
    )
    plt.title('Age vs Average Spending Score Trend')
    plt.savefig('artifacts/plots/age_spending_trend.png')
    plt.close()


# 7. Correlation Heatmap
def correlation_heatmap(df):
    corr = df[
        [
            'Age',
            'Annual Income ($)',
            'Spending Score (1-100)',
            'Work Experience',
            'Family Size'
        ]
    ].corr()

    plt.figure(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('artifacts/plots/correlation_heatmap.png')
    plt.close()


# 8. Gender Distribution Pie Chart
def gender_distribution(df):
    plt.figure(figsize=(5, 5))
    df['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.savefig('artifacts/plots/gender_distribution.png')
    plt.close()


# 9. Average Work Experience by Gender
def work_experience_gender(df):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='Gender',
        y='Work Experience',
        data=df
    )
    plt.title('Average Work Experience by Gender')
    plt.savefig('artifacts/plots/work_experience_gender.png')
    plt.close()


# 10. Age vs Average Spending Score Area Plot
def spending_score_area(df):
    plt.figure(figsize=(8, 5))
    df.groupby('Age')['Spending Score (1-100)'].mean().plot(
        kind='area',
        alpha=0.5
    )
    plt.title('Age vs Average Spending Score Area Plot')
    plt.savefig('artifacts/plots/spending_score_area.png')
    plt.close()


# 11. Annual Income by Family Size Violin Plot
def income_family_violin(df):
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x='Family Size',
        y='Annual Income ($)',
        data=df
    )
    plt.title('Annual Income by Family Size')
    plt.savefig('artifacts/plots/income_family_violin.png')
    plt.close()


# 12. Actual vs Predicted
def actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Spending Score")
    plt.ylabel("Predicted Spending Score")
    plt.title("Actual vs Predicted")
    plt.savefig('artifacts/plots/actual_vs_predicted.png')
    plt.close()
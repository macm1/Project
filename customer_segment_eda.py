# customer_segment_eda.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import numpy as np

def perform_eda(file_path):
    # Read the dataset
    df = pd.read_excel(file_path)

    # Calculate customer lifetime
    df['customer_lifetime'] = 2024 - df['Dt_Customer'].dt.year
    df = df.drop(['Dt_Customer'], axis=1)  # drop the column

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df['Income'] = imputer.fit_transform(df[['Income']])

    # Extracting year, month, and day from Dt_Customer
    df['age'] = 2024 - df["Year_Birth"]  # create the age column
    df = df.drop(['Year_Birth'], axis=1)  # drop the column
    df.drop(['Z_CostContact', 'Z_Revenue'], axis=1)  # low variance in distribution

    # Drop unnecessary columns
    df["Marital_Status"] = df["Marital_Status"].replace({"Married": "Married", "Together": "Married", "Absurd": "Single", 'YOLO': "Single", 'Alone': "Single", 'single': "Single"})

    # Map education categories to numerical values
    education_mapping = {
        'Basic': 1,
        '2n Cycle': 2,
        'Graduation': 3,
        'Master': 4,
        'PhD': 5
    }
    df['Education'] = df['Education'].map(education_mapping)

    # Perform one-hot encoding for Marital_Status
    one_hot_encoded = pd.get_dummies(df, columns=['Marital_Status'])
    df = pd.concat([df, one_hot_encoded], axis=1)
    df.drop('Marital_Status', axis=1, inplace=True)
    df['Spent'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].apply(lambda x: np.sum(x.dropna()), axis=1)
    #df['Spent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    print(df[['Spent', 'Income']].isnull().sum())

    # Drop outliers
    df1=df.copy()
    # Scale the data
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(scaled_df)
    pca_ds = pd.DataFrame(pca.transform(scaled_df), columns=["PC1", "PC2", "PC3"])

    return pca_ds,df1

# Perform EDA
if __name__ == "__main__":
    pca_ds,df1 = perform_eda(r'C:\Users\Riddhima\Downloads\marketing_campaign.xlsx')
    with open('eda.pkl', 'wb') as f:
        pickle.dump(pca_ds, f)
    with open('dataframe.pkl', 'wb') as f:
        pickle.dump(df1,f)    

    with open('eda.pkl', 'rb') as f:
        pca_ds= pickle.load(f)
        print(pca_ds)
    with open('dataframe.pkl', 'rb') as f:
        df1= pickle.load(f)
        print(df1) 
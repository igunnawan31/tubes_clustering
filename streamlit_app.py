import streamlit as st
import pandas as pd
import io
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Title and Information
st.title('🎈 Clustering with Gym Dataset')
st.info('Website for Machine Learning Model')

# Sidebar Parent Options
st.sidebar.header("Main Menu")
parentoption = st.sidebar.radio("Select Section:", 
                                ['Exploratory Data Analysis', 'Models', 'Input Data'])

# Load Dataset
data = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')

# 1. Exploratory Data Analysis
if parentoption == 'Exploratory Data Analysis':
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    # Sidebar Child Options
    option = st.sidebar.radio("Choose EDA Section:", 
                              ['Dataset Overview', 'Feature Data', 'PCA Data', 'Data Normalization'])

    # Option 1: Dataset Overview
    if option == 'Dataset Overview':
        st.subheader("📊 Dataset Overview")
        with st.expander('Dataset Preview'):
            st.write("### Dataset Preview:")
            st.dataframe(data.head())

            st.write("### Data Information:")
            buffer = io.StringIO()
            data.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

    # Option 2: Feature Data
    elif option == 'Feature Data':
        st.subheader("📈 Feature Data")
        with st.expander('Feature Data'):
            features = ['Calories_Burned', 'Water_Intake (liters)', 
                        'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
            st.write("### Selected Features:")
            feature_data = data[features]
            st.dataframe(feature_data.head())

            st.write("### Other Data:")
            restdata = data.drop(features, axis=1)
            st.dataframe(restdata.head())

    # Option 3: PCA Data
    elif option == 'PCA Data':
        st.subheader("🧩 PCA Data")
        with st.expander('PCA Data'):
            features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']
            st.write("### Features for PCA:")
            datapca = data[features_to_pca]
            st.dataframe(datapca.head())

            st.write("### Cleaned PCA Data:")
            datapca_clean = datapca.fillna(datapca.mean())
            st.dataframe(datapca_clean.head())

            st.write("### Normalization PCA Data:")
            scaler = StandardScaler()
            PCA_scaled = scaler.fit_transform(datapca_clean)
            PCA_scaled_df = pd.DataFrame(PCA_scaled, columns=features_to_pca)

            st.write("#### PCA Data after Cleaned & Normalization (Standard Scaler):")
            st.dataframe(PCA_scaled_df.head())

    # Option 4: Data Normalization
    elif option == 'Data Normalization':
        st.subheader("🔧 Data Normalization")
        with st.expander('Data Normalization'):
            st.write("### Feature Data after Handling Missing Values:")
            features = ['Calories_Burned', 'Water_Intake (liters)', 
                        'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
            feature_data = data[features]
            feature_data_clean = feature_data.fillna(feature_data.mean())
            st.dataframe(feature_data_clean.head())

            st.write("### Feature Data Types:")
            st.write(feature_data_clean.dtypes)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feature_data_clean)
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_data.columns)

            st.write("### Feature Data after Normalization (StandardScaler):")
            st.dataframe(X_scaled_df.head())

            # Adding PCA to normalized data
            features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']
            datapca = data[features_to_pca]
            datapca_clean = datapca.fillna(datapca.mean())

            scaler = StandardScaler()
            PCA_scaled = scaler.fit_transform(datapca_clean)
            PCA_scaled_df = pd.DataFrame(PCA_scaled, columns=features_to_pca)

            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(PCA_scaled_df)
            df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

            cleaned_data = pd.concat([X_scaled_df, df_pca], axis=1)
            scaler_data = StandardScaler()
            scaled_data = scaler.fit_transform(cleaned_data)
            scaled_data_final = pd.DataFrame(scaled_data, columns=cleaned_data.columns)

            st.write("### Data Final for Clustering")
            st.dataframe(scaled_data_final.head())

# 2. Models Section
elif parentoption == 'Models':
    st.subheader("📊 Models for Clustering")

    # Sidebar for Model Options
    st.sidebar.subheader("KMeans Settings")
    k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)

    st.write("### Data Preparation for Clustering")

    st.dataframe(scaled_data_final.head())

    features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']
    datapca = data[features_to_pca].fillna(data[features_to_pca].mean())

    scaler = StandardScaler()
    scaled_pca = scaler.fit_transform(datapca)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_pca)
  
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    st.write("### PCA-Reduced Data:")
    st.dataframe(df_pca.head())

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca)

    st.write(f"### KMeans Clustering with {k_clusters} Clusters:")
    st.dataframe(df_pca.head())

    # Visualization
    st.write("### Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, legend="full", ax=ax)
    plt.title("KMeans Clustering Results")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(fig)

# 3. Input Data Section
elif parentoption == 'Input Data':
    st.subheader("📥 Input Data for Clustering")
    st.write("### Upload Your Own Data:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.dataframe(user_data.head())

        st.write("### Scaling Uploaded Data:")
        scaler = StandardScaler()
        scaled_user_data = scaler.fit_transform(user_data.fillna(0))
        scaled_user_df = pd.DataFrame(scaled_user_data, columns=user_data.columns)

        st.write("### Scaled Data:")
        st.dataframe(scaled_user_df.head())

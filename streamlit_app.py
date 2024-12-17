import streamlit as st
import pandas as pd
import numpy as np
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
data_url = 'https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv'
data = pd.read_csv(data_url)

# Features for processing
features = ['Calories_Burned', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']

# Cleaning and Scaling Data
feature_data_clean = data[features].fillna(data[features].mean())
datapca_clean = data[features_to_pca].fillna(data[features_to_pca].mean())

scaler_features = StandardScaler()
scaler_pca = StandardScaler()

X_scaled_features = scaler_features.fit_transform(feature_data_clean)
X_scaled_pca = scaler_pca.fit_transform(datapca_clean)

# PCA Transformation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled_pca)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Combine Data for Clustering
combined_data = np.hstack([X_scaled_features, pca_result])
combined_columns = features + ['PC1', 'PC2']
cleaned_data = pd.DataFrame(combined_data, columns=combined_columns)

# Default Clustering with 3 Clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(combined_data)

# 1. Exploratory Data Analysis
if parentoption == 'Exploratory Data Analysis':
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    st.write("### Cleaned Feature Data:")
    st.dataframe(feature_data_clean.head())

    st.write("### PCA-Reduced Data:")
    st.dataframe(df_pca.head())

    # Visualization
    st.write("### PCA Visualization with Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=cleaned_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
    plt.title("PCA and KMeans Clustering")
    st.pyplot(fig)

# 2. Models Section
elif parentoption == 'Models':
    st.subheader("📊 KMeans Clustering Model")
    k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)

    # Fit KMeans with User-Selected Clusters
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    cleaned_data['Cluster'] = kmeans.fit_predict(combined_data)

    st.write(f"### Clustered Data with {k_clusters} Clusters:")
    st.dataframe(cleaned_data.head())

    # Visualization
    st.write(f"### KMeans Clustering Visualization with {k_clusters} Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=cleaned_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
    plt.title(f"KMeans Clustering Results with {k_clusters} Clusters")
    st.pyplot(fig)

# 3. Input Data Section
elif parentoption == 'Input Data':
    st.subheader("📥 Input Data for Clustering")
    st.write("### Enter Your Data Below:")

    # User Input Form
    with st.form("user_input_form"):
        calories_burned = st.slider('Calories Burned', 0, 1000, 500)
        water_intake = st.slider('Water Intake (liters)', 0.0, 5.0, 2.5)
        workout_frequency = st.slider('Workout Frequency (days/week)', 0, 7, 3)
        fat_percentage = st.slider('Fat Percentage (%)', 0.0, 50.0, 25.0)
        bmi = st.slider('BMI', 0.0, 50.0, 22.5)
        
        weight = st.slider('Weight (kg)', 30.0, 150.0, 70.0)
        height = st.slider('Height (m)', 1.0, 2.5, 1.75)
        max_bpm = st.slider('Max BPM', 50, 200, 120)
        avg_bpm = st.slider('Avg BPM', 50, 200, 100)
        resting_bpm = st.slider('Resting BPM', 40, 100, 60)
        experience_level = st.slider('Experience Level (1-5)', 1, 5, 3)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Prepare user input
        user_features = np.array([[calories_burned, water_intake, workout_frequency, fat_percentage, bmi]])
        user_pca_input = np.array([[weight, height, max_bpm, avg_bpm, resting_bpm, experience_level]])

        # Scale and Transform Input
        user_scaled_features = scaler_features.transform(user_features)
        user_scaled_pca = scaler_pca.transform(user_pca_input)
        user_pca_result = pca.transform(user_scaled_pca)

        user_combined = np.hstack([user_scaled_features, user_pca_result])
        user_cluster = kmeans.predict(user_combined)[0]

        # Display Prediction
        st.success(f"Your data is predicted to belong to **Cluster {user_cluster}**!")

        # Visualization
        st.write("### Your Data Point in the Cluster Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=cleaned_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
        plt.scatter(user_pca_result[0, 0], user_pca_result[0, 1], color='red', s=200, label='Your Input')
        plt.title("Your Data in the PCA Cluster Space")
        plt.legend()
        st.pyplot(fig)

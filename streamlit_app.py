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
features = ['Calories_Burned', 'Water_Intake (liters)', 
            'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']

# Fill missing values and scale data
feature_data_clean = data[features].fillna(data[features].mean())
datapca_clean = data[features_to_pca].fillna(data[features_to_pca].mean())

# Scalers and PCA setup
scaler_features = StandardScaler().fit(feature_data_clean)
scaler_pca = StandardScaler().fit(datapca_clean)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaler_pca.transform(scaler_pca))

# Combined data for clustering
X_combined = np.hstack([
    scaler_features.transform(feature_data_clean), 
    pca_result
])
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_combined)

# Final DataFrame for visualization
clustered_data = pd.DataFrame(X_combined, columns=features + ['PC1', 'PC2'])
clustered_data['Cluster'] = kmeans.labels_

# 1. Exploratory Data Analysis
if parentoption == 'Exploratory Data Analysis':
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    st.write("### Selected Features:")
    st.dataframe(feature_data_clean.head())

    st.write("### PCA Components:")
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    st.dataframe(pca_df.head())

    st.write("### Clustered Data:")
    st.dataframe(clustered_data.head())

    # Visualization
    st.write("### PCA Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=clustered_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
    plt.title("PCA and KMeans Clustering")
    st.pyplot(fig)

# 2. Models Section
elif parentoption == 'Models':
    st.subheader("📊 KMeans Clustering Model")
    k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)

    # KMeans Refit
    kmeans = KMeans(n_clusters=k_clusters, random_state=42).fit(X_combined)
    clustered_data['Cluster'] = kmeans.labels_

    st.write("### Clustered Data:")
    st.dataframe(clustered_data.head())

    # Visualization
    st.write(f"### KMeans Clustering Visualization with {k_clusters} Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=clustered_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
    plt.title("KMeans Clustering Results")
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

        # Submit Button
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Collect user input
        user_features = pd.DataFrame([[calories_burned, water_intake, workout_frequency, fat_percentage, bmi]],
                                     columns=features)
        user_pca_input = pd.DataFrame([[weight, height, max_bpm, avg_bpm, resting_bpm, experience_level]],
                                      columns=features_to_pca)

        # Transform user input
        user_scaled_features = scaler_features.transform(user_features)
        user_scaled_pca = scaler_pca.transform(user_pca_input)
        user_pca_result = pca.transform(user_scaled_pca)

        # Combine user input
        user_combined = np.hstack([user_scaled_features, user_pca_result])
        user_cluster = kmeans.predict(user_combined)[0]

        # Display results
        st.write("### Cluster Prediction Result:")
        st.success(f"Your data is predicted to belong to **Cluster {user_cluster}**!")

        # Visualize user input on scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=clustered_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
        plt.scatter(user_pca_result[0, 0], user_pca_result[0, 1], color='red', s=200, label='Your Input')
        plt.title("Your Data Point in KMeans Clustering")
        plt.legend()
        st.pyplot(fig)

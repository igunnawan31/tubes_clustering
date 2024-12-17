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
  
    features = ['Calories_Burned', 'Water_Intake (liters)', 
                'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
    feature_data = data[features]
    feature_data_clean = feature_data.fillna(feature_data.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_data_clean)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_data.columns)

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

    scaler = StandardScaler()
    scaled_pca = scaler.fit_transform(scaled_data_final)
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
    st.write("### Enter Your Own Data:")

    with st.form(key='user_input_form'):
        # User Input Data
        age = st.slider('Age', 0, 100, 30)
        gender = st.radio('Gender', ('Male', 'Female'))
        weight = st.slider('Weight (kg)', 30.0, 150.0, 70.0)
        height = st.slider('Height (m)', 1.0, 2.5, 1.75)
        max_bpm = st.slider('Max BPM', 50, 250, 180)
        avg_bpm = st.slider('Average BPM', 50, 250, 140)
        resting_bpm = st.slider('Resting BPM', 30, 100, 60)
        session_duration = st.slider('Session Duration (hours)', 0.0, 5.0, 1.5)
        calories_burned = st.slider('Calories Burned', 50, 1000, 400)
        workout_type = st.radio('Workout Type', ('Yoga', 'HIIT', 'Cardio', 'Strength'))
        fat_percentage = st.slider('Fat_Percentage', 5.0, 50.0, 25.0)
        water_intake = st.slider('Water Intake (liters)', 0.0, 10.0, 2.5)
        workout_frequency = st.slider('Workout_Frequency (days/week)', 1, 7, 3)
        experience_level = st.slider('Experience Level (0 = Beginner, 5 = Expert)', 0, 5, 2)
        bmi = st.slider('BMI', 10.0, 50.0, 22.0)

        # Form submit button inside the form block
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Collect User Input Data into a Dictionary
        user_input_data = {
            'Age': age,
            'Gender': gender,
            'Weight (kg)': weight,
            'Height (m)': height,
            'Max_BPM': max_bpm,
            'Avg_BPM': avg_bpm,
            'Resting_BPM': resting_bpm,
            'Session Duration (hours)': session_duration,
            'Calories_Burned': calories_burned,
            'Workout Type': workout_type,
            'Fat_Percentage': fat_percentage,
            'Water_Intake (liters)': water_intake,
            'Workout_Frequency (days/week)': workout_frequency,
            'Experience_Level': experience_level,
            'BMI': bmi
        }

        # Convert User Input to DataFrame
        input_df = pd.DataFrame([user_input_data])
        st.dataframe(input_df.head())

        # KMeans Model Preparation (Training)
        features = ['Calories_Burned', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
        features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']

        # Train with cleaned data
        feature_data_clean = data[features].fillna(data[features].mean())
        scaler_features = StandardScaler()
        X_scaled = scaler_features.fit_transform(feature_data_clean)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_data_clean.columns)

        # PCA for other features
        datapca = data[features_to_pca].fillna(data[features_to_pca].mean())
        scaler_pca = StandardScaler()
        PCA_scaled = scaler_pca.fit_transform(datapca)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(PCA_scaled)
        df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

        # Combine and scale data for training
        cleaned_data = pd.concat([X_scaled_df, df_pca], axis=1)
        scaler_combined = StandardScaler()
        scaled_data = scaler_combined.fit_transform(cleaned_data)
        scaled_data_df = pd.DataFrame(scaled_data, columns=cleaned_data.columns)

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_data_df)

        # Prediction on User Input
        try:
            # Extract user input features
            user_input_features = input_df[features]
            user_input_pca = input_df[features_to_pca]

            # Scale user input features using trained scalers
            user_input_scaled_features = scaler_features.transform(user_input_features.fillna(user_input_features.mean()))
            user_input_scaled_pca = scaler_pca.transform(user_input_pca.fillna(user_input_pca.mean()))

            # Apply PCA to user input
            user_input_pca = pca.transform(user_input_scaled_pca)

            # Combine user input features and PCA components
            user_input_combined = pd.concat(
                [pd.DataFrame(user_input_scaled_features, columns=features),
                 pd.DataFrame(user_input_pca, columns=['PC1', 'PC2'])],
                axis=1
            )

            # Scale combined features
            user_input_final = scaler_combined.transform(user_input_combined)

            # Predict cluster
            user_cluster = kmeans.predict(user_input_final)[0]
            user_input_combined['Cluster'] = user_cluster

            # Display the result
            st.write("### Your Input Data with Predicted Cluster:")
            st.dataframe(user_input_combined)
            st.write(f"### Predicted Cluster: {user_cluster}")

            # Visualization
            st.write("### Cluster Visualization with Your Data:")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=scaled_data_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, legend="full", ax=ax)
            plt.scatter(user_input_combined['PC1'], user_input_combined['PC2'], color='red', s=200, label='Your Input')
            plt.title("KMeans Clustering with User Data")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            st.pyplot(fig)

        except ValueError as e:
            st.error(f"An error occurred: {e}")

import streamlit as st
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler

# Title and Information
st.title('🎈 Clustering with Gym Dataset')
st.info('Website for Machine Learning Model')

# Sidebar Options
st.sidebar.header("Options Menu")
option = st.sidebar.radio("Choose a Section:", 
                          ['Dataset Overview', 'Feature Data', 'PCA Data', 'Data Normalization'])

# Load Dataset
data = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')

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

        st.write("### PCA Data after Cleaned & Normalization (Standard Scaler):")
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

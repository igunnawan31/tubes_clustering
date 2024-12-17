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
@st.cache_data
def load_data():
    data_url = 'https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv'
    return pd.read_csv(data_url)

data = load_data()

# Function to preprocess and return scaled data
def preprocess_data(data):
    features = ['Calories_Burned', 'Water_Intake (liters)', 
                'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
    features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 
                       'Avg_BPM', 'Resting_BPM', 'Experience_Level']

    # Fill missing values
    feature_data = data[features].fillna(data[features].mean())
    datapca = data[features_to_pca].fillna(data[features_to_pca].mean())

    # Scaling
    scaler_features = StandardScaler()
    X_scaled = scaler_features.fit_transform(feature_data)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    scaler_pca = StandardScaler()
    PCA_scaled = scaler_pca.fit_transform(datapca)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(PCA_scaled)

    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

    # Final combined dataset
    final_data = pd.concat([X_scaled_df, df_pca], axis=1)
    return final_data, scaler_features, scaler_pca, pca

final_data, scaler_features, scaler_pca, pca = preprocess_data(data)

# 1. Exploratory Data Analysis
if parentoption == 'Exploratory Data Analysis':
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    option = st.sidebar.radio("Choose EDA Section:", 
                              ['Dataset Overview', 'Feature Data', 'PCA Data'])

    if option == 'Dataset Overview':
        st.subheader("📊 Dataset Overview")
        with st.expander('Dataset Preview'):
            st.dataframe(data.head())

        st.write("### Data Information:")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

    elif option == 'Feature Data':
        st.subheader("📈 Feature Data")
        features = ['Calories_Burned', 'Water_Intake (liters)', 
                    'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
        st.write("### Selected Features:")
        st.dataframe(data[features].head())

    elif option == 'PCA Data':
        st.subheader("🧩 PCA Data")
        st.write("### PCA-Reduced Data:")
        st.dataframe(final_data[['PC1', 'PC2']].head())

# 2. Models Section
elif parentoption == 'Models':
    st.subheader("📊 Models for Clustering")

    st.sidebar.subheader("KMeans Settings")
    k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    final_data['Cluster'] = kmeans.fit_predict(final_data)

    st.write("### Clustered Data:")
    st.dataframe(final_data.head())

    st.write("### Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=final_data, x='PC1', y='PC2', hue='Cluster', 
                    palette='Set2', s=100, legend="full", ax=ax)
    plt.title("KMeans Clustering Results")
    st.pyplot(fig)

# 3. Input Data Section
elif parentoption == 'Input Data':
    st.subheader("📥 Input Data for Clustering")
    with st.form("user_input_form"):
        # User Input
        age = st.slider('Age', 0, 100, 30)
        weight = st.slider('Weight (kg)', 30.0, 150.0, 70.0)
        height = st.slider('Height (m)', 1.0, 2.5, 1.75)
        max_bpm = st.slider('Max BPM', 50, 250, 180)
        avg_bpm = st.slider('Average BPM', 50, 250, 140)
        resting_bpm = st.slider('Resting BPM', 30, 100, 60)
        calories_burned = st.slider('Calories Burned', 50, 1000, 400)
        fat_percentage = st.slider('Fat Percentage (%)', 5.0, 50.0, 25.0)
        water_intake = st.slider('Water Intake (liters)', 0.0, 10.0, 2.5)
        workout_frequency = st.slider('Workout Frequency (days/week)', 1, 7, 3)
        experience_level = st.slider('Experience Level (0 = Beginner, 5 = Expert)', 0, 5, 2)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Create user data
        user_data = pd.DataFrame({
            'Calories_Burned': [calories_burned],
            'Water_Intake (liters)': [water_intake],
            'Workout_Frequency (days/week)': [workout_frequency],
            'Fat_Percentage': [fat_percentage],
            'BMI': [weight / (height ** 2)],
            'Weight (kg)': [weight],
            'Height (m)': [height],
            'Max_BPM': [max_bpm],
            'Avg_BPM': [avg_bpm],
            'Resting_BPM': [resting_bpm],
            'Experience_Level': [experience_level]
        })

        # Scaling user data
        user_features = scaler_features.transform(user_data[['Calories_Burned', 'Water_Intake (liters)',
                                                             'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']])
        user_pca = pca.transform(scaler_pca.transform(user_data[['Weight (kg)', 'Height (m)', 'Max_BPM', 
                                                                'Avg_BPM', 'Resting_BPM', 'Experience_Level']]))

        # Combine user data
        user_final = pd.concat([pd.DataFrame(user_features, columns=['Calories_Burned', 'Water_Intake (liters)', 
                                                                     'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']),
                                pd.DataFrame(user_pca, columns=['PC1', 'PC2'])], axis=1)

        # Predict cluster
        user_cluster = kmeans.predict(user_final)[0]
        st.success(f"Predicted Cluster: {user_cluster}")

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=final_data, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=100, ax=ax)
        plt.scatter(user_final['PC1'], user_final['PC2'], color='red', s=200, label='Your Data')
        plt.title("KMeans Clustering with Your Data")
        plt.legend()
        st.pyplot(fig)

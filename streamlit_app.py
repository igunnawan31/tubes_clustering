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
                              ['Dataset Overview', 'Feature Data', 'Data Normalization'])

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
            features = ['Calories_Burned','Fat_Percentage', 'Session_Duration (hours)', 'Experience_Level']
            st.write("### Selected Features:")
            feature_data = data[features]
            st.dataframe(feature_data.head())

            st.write("### Other Data:")
            restdata = data.drop(features, axis=1)
            st.dataframe(restdata.head())

    # Option 3: Data Normalization
    elif option == 'Data Normalization':
        st.subheader("🔧 Data Normalization")
        with st.expander('Data Normalization'):
            st.write("### Feature Data after Handling Missing Values:")
            WorkoutTypeMap = {
                "Strength": 0,
                "Cardio" : 1,
                "Yoga" : 2,
                "HIIT" : 3
            }
            
            GenderMap = {
                "Male" : 0,
                "Female" : 1
            }
            data['Gender'] = data['Gender'].map(GenderMap)
            data['Workout_Type'] = data['Workout_Type'].map(WorkoutTypeMap)
            
            for feature in data.columns:
                # Hitung Q1, Q3, dan IQR using the DataFrame
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
        
                lower_bound = Q1 - 1.0 * IQR
                upper_bound = Q3 + 1.0 * IQR
            
                outliers = (data[feature] < lower_bound) | (data[feature] > upper_bound)
            
                print(f"\nFeature: {feature}")
                print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
                print(f"Number of Outliers: {outliers.sum()}")
            
                X_scaled_df = data[~outliers]
          
                print("\nNew Shape: ", X_scaled_df.shape)
                cleaned_data = X_scaled_df

            features = ['Calories_Burned','Fat_Percentage', 'Session_Duration (hours)', 'Experience_Level']
            cleaned_data = data[features]
          
            st.dataframe(cleaned_data.head())

            st.write("### Feature Data Types:")
            st.write(cleaned_data.dtypes)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cleaned_data)
            X_scaled_df = pd.DataFrame(X_scaled, columns=cleaned_data.columns)

            st.write("### Feature Data after Normalization (StandardScaler):")
            st.dataframe(X_scaled_df.head())

# 2. Models Section
elif parentoption == 'Models':
    st.subheader("📊 Models for Clustering")

    # Sidebar for Model Options
    st.sidebar.subheader("KMeans Settings")
    k_clusters = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 3)
  
    WorkoutTypeMap = {
        "Strength": 0,
        "Cardio" : 1,
        "Yoga" : 2,
        "HIIT" : 3
    }
    
    GenderMap = {
        "Male" : 0,
        "Female" : 1
    }
    data['Gender'] = data['Gender'].map(GenderMap)
    data['Workout_Type'] = data['Workout_Type'].map(WorkoutTypeMap)
    
    for feature in data.columns:
        # Hitung Q1, Q3, dan IQR using the DataFrame
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.0 * IQR
        upper_bound = Q3 + 1.0 * IQR
    
        outliers = (data[feature] < lower_bound) | (data[feature] > upper_bound)
    
        print(f"\nFeature: {feature}")
        print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"Number of Outliers: {outliers.sum()}")
    
        X_scaled_df = data[~outliers]
  
        print("\nNew Shape: ", X_scaled_df.shape)
        cleaned_data = X_scaled_df

    features = ['Calories_Burned','Fat_Percentage', 'Session_Duration (hours)', 'Experience_Level']
    cleaned_data = data[features]
  
    st.dataframe(cleaned_data.head())

    st.write("### Feature Data Types:")
    st.write(cleaned_data.dtypes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cleaned_data)
    X_scaled_df = pd.DataFrame(X_scaled, columns=cleaned_data.columns)

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled_df)
    X_scaled_df['Cluster'] = labels

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled_df.drop(columns=['Cluster']))
    pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    for cluster in range(k_clusters):
        cluster_data = pca_df[pca_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', cmap='viridis', s=100, edgecolors='k')
    
    cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
                c='red', marker='X', s=300, label='Centroids')
    
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Visualisasi Hasil Clustering dengan PCA')
    plt.legend()
    
    st.pyplot(plt)

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
        session_duration = st.slider('Session_Duration (hours)', 0.0, 5.0, 1.5)
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
            'Session_Duration (hours)': session_duration,
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

        WorkoutTypeMap = {
            "Strength": 0,
            "Cardio" : 1,
            "Yoga" : 2,
            "HIIT" : 3
        }
        
        GenderMap = {
            "Male" : 0,
            "Female" : 1
        }
        data['Gender'] = data['Gender'].map(GenderMap)
        data['Workout_Type'] = data['Workout_Type'].map(WorkoutTypeMap)
        
        for feature in data.columns:
            # Hitung Q1, Q3, dan IQR using the DataFrame
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
    
            lower_bound = Q1 - 1.0 * IQR
            upper_bound = Q3 + 1.0 * IQR
        
            outliers = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        
            print(f"\nFeature: {feature}")
            print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
            print(f"Number of Outliers: {outliers.sum()}")
        
            X_scaled_df = data[~outliers]
      
            print("\nNew Shape: ", X_scaled_df.shape)
            cleaned_data = X_scaled_df
    
        features = ['Calories_Burned','Fat_Percentage', 'Session_Duration (hours)', 'Experience_Level']
        cleaned_data = data[features]
      
        st.dataframe(cleaned_data.head())
    
        st.write("### Feature Data Types:")
        st.write(cleaned_data.dtypes)
    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cleaned_data)
        X_scaled_df = pd.DataFrame(X_scaled, columns=cleaned_data.columns)
    
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X_scaled_df)
        X_scaled_df['Cluster'] = labels
    
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled_df.drop(columns=['Cluster']))
        pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = labels

        # Predict Cluster for User Input
        user_input_scaled = scaler.transform(input_df[features])  # Match features used for scaling
        user_input_pca = pca.transform(user_input_scaled)  # Project scaled data into PCA space

        predicted_cluster = kmeans.predict(user_input_scaled)[0] 
        
        # Visualization
        plt.figure(figsize=(10, 6))
        for cluster in range(2):  # Adjust this range based on your number of clusters
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', s=100, edgecolors='k')
        
        # Cluster centroids
        cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
                    c='red', marker='X', s=300, label='Centroids')
        
        # Add user input point
        plt.scatter(user_input_pca[0, 0], user_input_pca[0, 1], 
                    c='blue', marker='o', s=200, label='User Input', edgecolors='k')
        
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('Visualization of Clustering with User Input')
        plt.legend()
        st.pyplot(plt)
        st.success(f"Your data belongs to Cluster: {predicted_cluster}")

import streamlit as st
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler

st.title('🎈 Clustering with Gym Dataset')
st.info('Website for Machine Learning Model')
data = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')

with st.expander('Data'):
    st.write("### Dataset Preview:")
    st.dataframe(data.head())  # Display the first few rows of the dataset

    st.write("### Data Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()  # Get the content of the buffer as a string
    st.text(info_str)

with st.expander('Feature Data'):
    st.write("### Feature Data:")
    features = ['Calories_Burned','Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Fat_Percentage', 'BMI']
    feature_data = data[features]
    st.dataframe(feature_data.head())
    
    st.write("### Data Lainnya:")
    restdata = data.drop(features, axis=1)
    st.dataframe(restdata.head())

    st.write("### PCA Data:")
    features_to_pca = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Experience_Level']
    datapca = restdata[features_to_pca]
    st.dataframe(datapca.head())

with st.expander('Data Normalization'):
    st.write("### Feature Data menggunakan StandardScaler (Setelah Normalisasi) :")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_data)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_data.columns)
    

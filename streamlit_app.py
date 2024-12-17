import streamlit as st
import pandas as pd
import io

st.title('🎈 Clustering with Gym Dataset')
st.info('Website for Machine Learning Model')

# Load Data
with st.expander('Data'):
    # Load the dataset
    data = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')

    st.write("### Dataset Preview:")
    st.dataframe(data.head())  # Display the first few rows of the dataset

    st.write("### Data Information:")
    buffer = io.StringIO()  # Create a string buffer
    data.info(buf=buffer)  # Write data.info() to the buffer
    info_str = buffer.getvalue()  # Get the content of the buffer as a string
    st.text(info_str)

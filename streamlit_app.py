import streamlit as st
import pandas as pd

st.title('🎈 Clustering with Gym Dataset')
st.info('Website for Machine Learning Modell')

with st.expander('Data') :
  data = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')
  st.write("### Dataset Preview:")
  st.dataframe(data.head())  # Display data preview

  st.write("### Data Information:")
  buffer = []
  data.info(buf=buffer.append)  # Capture the output of data.info()
  info_str = "\n".join(buffer)  # Join the captured output as a single string
  st.text(info_str)

  

import streamlit as st
import pandas as pd

st.title('🎈 Clustering with Gym Dataset')
st.info('Website for Machine Learning Modell')

with st.expander('Data') :
  data = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')
  data
  data.info()

  

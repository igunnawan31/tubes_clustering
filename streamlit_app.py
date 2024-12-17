import streamlit as st
import pandas as pd

st.title('🎈 Clustering with Gym Dataset')
st.Info('Website for Machine Learning Modell')
df = pd.read_csv('https://raw.githubusercontent.com/igunnawan31/data/refs/heads/main/gym_members_exercise_tracking.csv')
df

import pandas as pd
import streamlit as st
from app.utils.preprocessing import clean_text

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data once"""
    data = pd.read_parquet("data/raw/data.parquet")
    data = data[['text', 'label']].dropna()
    data['text'] = data['text'].apply(clean_text)
    return data
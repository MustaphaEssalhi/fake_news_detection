import os
import tensorflow as tf
import pickle
import streamlit as st

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer once and cache them"""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'lstm-last-5.h5')
        tokenizer_path = os.path.join(base_path, 'tokenizer-last-5.pickle')

        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop()

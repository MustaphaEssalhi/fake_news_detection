import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.load_model import load_model_and_tokenizer
from data.load_data import load_and_preprocess_data
from app.utils.preprocessing import clean_text, preprocess_text
from app.utils.visualization import plot_confusion_matrix, plot_roc_curve

def compute_metrics(model, tokenizer, X_test, y_test, max_len=200):
    """Compute evaluation metrics"""
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    metrics = model.evaluate(X_test_pad, y_test, verbose=0)
    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = metrics[1]
    precision = metrics[2]
    recall = metrics[3]
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    roc_auc = plot_roc_curve(y_test, y_pred_probs)
    plot_confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc
    }

def main():
    st.set_page_config(page_title="Fake News Detection", layout="wide")
    st.title("📰 Fake News Detection App")
    st.markdown("This app uses a trained LSTM model to classify news articles.")

    try:
        model, tokenizer = load_model_and_tokenizer()
        data = load_and_preprocess_data()
    except Exception as e:
        st.error(f"Failed to load model or data: {e}")
        st.stop()

    # Split data
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute metrics
    metrics = compute_metrics(model, tokenizer, X_test, y_test)

    # Display metrics
    st.subheader("📊 Model Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Value': [f"{metrics['accuracy']:.4f}", 
                  f"{metrics['precision']:.4f}", 
                  f"{metrics['recall']:.4f}", 
                  f"{metrics['f1_score']:.4f}", 
                  f"{metrics['roc_auc']:.4f}"]
    })
    st.table(metrics_df)

    # Display visualizations
    st.subheader("📈 Evaluation Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.image('reports/figures/confusion_matrix.png')
    with col2:
        st.image('reports/figures/roc_curve.png')

    # Prediction section
    st.subheader("📝 News Classification")

    col_fake, col_real = st.columns([1, 1])
    if "article_input" not in st.session_state:
        st.session_state.article_input = ""

    if col_fake.button("Use Sample Fake News"):
        st.session_state.article_input = (
            "BREAKING: Government confirms aliens built the pyramids. "
            "Anonymous officials revealed this morning that extraterrestrial beings had full control over ancient Egypt."
        )

    if col_real.button("Use Sample Real News"):
        st.session_state.article_input = (
            "The central bank announced a reduction in interest rates today to combat inflation pressures, "
            "marking a shift in monetary policy for the upcoming quarter."
        )

    user_input = st.text_area("Enter news article:", value=st.session_state.article_input, height=200)

    if st.button("🔍 Predict") and user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                padded_input = preprocess_text(user_input, tokenizer)
                prediction = model.predict(padded_input, verbose=0)[0][0]
                label = "Fake" if prediction > 0.5 else "Real"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                st.subheader("🧾 Result")
                st.write(f"**Prediction:** {label}")
                st.metric(label="Confidence", value=f"{confidence:.2%}")
                st.progress(float(confidence))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 14px;'>"
        "This project is made by <strong>ESSALHI MUSTAPHA, CHAFIK MOHAMED, HAYTAM GHOUTANI, CHADI HUSSEIN</strong>"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()


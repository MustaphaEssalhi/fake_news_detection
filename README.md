# Fake News Detection App

This project is a **Fake News Detection** app built using **Streamlit**, **TensorFlow**, and **NLP** techniques. The app utilizes a trained LSTM model to classify news articles as **Fake** or **Real** based on their content.

## Features

* **Fake News Detection**: Classifies news articles as fake or real.
* **Model Evaluation**: Displays performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
* **Interactive Visualization**: Displays confusion matrix and ROC curve.
* **Sample Articles**: Generate fake or real sample news articles with the press of a button.

## Tech Stack

* **Frontend**: Streamlit for building the user interface.
* **Backend**: TensorFlow for machine learning model.
* **Data**: Uses preprocessed text data for training and testing.
* **Visualization**: Matplotlib and Seaborn for plotting evaluation results.

## Requirements

To install the necessary libraries and dependencies for this project, create a virtual environment and install the packages from the `requirements.txt` file:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On MacOS/Linux
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```txt
streamlit==1.28.0
tensorflow==2.12.0
pandas==2.0.3
numpy==1.24.3
nltk==3.8.1
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
pyarrow==12.0.1
```

## Data

The dataset (`data.parquet`) is required to run the app but is not included in the repository due to GitHub's file size limitations.

You can download it using the following Python snippet:

```python
import pandas as pd

url = "hf://datasets/davanstrien/WELFake/data/train-00000-of-00001-290868f0a36350c5.parquet"

df = pd.read_parquet(url, engine='pyarrow')

df.to_parquet("data/raw/data.parquet")



## How to Run the Application

1. Clone the repository:

```bash
git clone <repository-url>
cd fake_news_detection
```

2. Set up the environment:

```bash
# Set up the virtual environment as shown above
```

3. Run the Streamlit app:

```bash
streamlit run app/main.py
```

This will start the app on your local machine, and you'll be able to interact with the **Fake News Detection** model via the browser.

## Project Structure

```plaintext
fake_news_detection
├── README.md                # Project documentation
├── app
│   ├── __init__.py
│   ├── main.py              # Main Streamlit app script
│   └── utils
│       ├── __init__.py
│       ├── preprocessing.py # Preprocessing utilities
│       └── visualization.py # Visualization utilities
├── data
│   ├── load_data.py         # Data loading and preprocessing
│   └── raw
│       └── data.parquet     # Raw dataset (for testing and training)
├── models
│   ├── __init__.py
│   ├── load_model.py        # Model loading function
│   ├── lstm-last-5.h5       # Trained LSTM model
│   └── tokenizer-last-5.pickle # Tokenizer for text processing
├── notebooks
│   └── lstm-last-version.ipynb # Jupyter notebook for training and experimentation
├── requirements.txt         # Python dependencies
├── setup.py                 # Setup script for packaging
└── tests
    ├── __init__.py
    └── test_preprocessing.py # Unit tests for preprocessing utilities
```

## Model

The project uses an **LSTM (Long Short-Term Memory)** model for text classification. The model is trained on a preprocessed text dataset to predict whether a news article is **Fake** or **Real**.

### Model Evaluation

* **Accuracy**: Measures the percentage of correct predictions.
* **Precision**: Measures the proportion of positive predictions that are actually correct.
* **Recall**: Measures the proportion of actual positives that are correctly identified.
* **F1-Score**: Harmonic mean of Precision and Recall.
* **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

## Authors

This project is made by:

* **ESSALHI MUSTAPHA**
* **CHAFIK MOHAMED**
* **HAYTAM GHOUTANI**
* **CHADI HUSSEIN**

## Next step
# TODO: Upgrade model to use Retrieval-Augmented Generation (RAG) with a pretrained BERT-based retriever and generator
# to improve fact-checking accuracy and provide evidence for fake/real news classification.
# Recommended setup: FAISS + DPR (Dense Passage Retrieval) for retrieval, and BART or T5 for generation.

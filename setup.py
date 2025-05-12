from setuptools import setup, find_packages

setup(
    name="fake_news_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'tensorflow>=2.12.0',
        'pandas>=2.0.3',
        'nltk>=3.8.1',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
        'pyarrow>=12.0.1'
    ],
    python_requires='>=3.8',
)
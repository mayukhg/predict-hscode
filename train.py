"""
Training script for HSCode prediction model.
This script loads the Kaggle dataset, preprocesses the data, trains a Logistic Regression model,
and saves the trained model and vectorizer for use by the API.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import preprocess_text, clean_hscode


def load_and_preprocess_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset and preprocess the text descriptions and HSCodes.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: Preprocessed features (X) and target labels (y)
    """
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    if 'item_description' not in df.columns or 'hscode' not in df.columns:
        raise ValueError("Dataset must contain 'item_description' and 'hscode' columns")
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Remove rows with missing values
    df = df.dropna(subset=['item_description', 'hscode'])
    print(f"After removing missing values: {len(df)} samples")
    
    # Preprocess text descriptions
    print("Preprocessing text descriptions...")
    df['cleaned_description'] = df['item_description'].apply(preprocess_text)
    
    # Clean and standardize HSCodes
    print("Cleaning HSCodes...")
    df['cleaned_hscode'] = df['hscode'].apply(clean_hscode)
    
    # Remove rows with empty descriptions after preprocessing
    df = df[df['cleaned_description'].str.len() > 0]
    print(f"After removing empty descriptions: {len(df)} samples")
    
    # Get unique HSCodes and their counts
    unique_hscodes = df['cleaned_hscode'].value_counts()
    print(f"Number of unique HSCodes: {len(unique_hscodes)}")
    print(f"Top 10 most common HSCodes:")
    print(unique_hscodes.head(10))
    
    # Prepare features and target
    X = df['cleaned_description']
    y = df['cleaned_hscode']
    
    return X, y


def train_model(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Train the Logistic Regression model with TF-IDF vectorization.
    
    Args:
        X: Text features
        y: Target HSCodes
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: Trained model, vectorizer, X_test, y_test
    """
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize TF-IDF vectorizer
    print("Initializing TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size for efficiency
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,            # Ignore terms that appear in less than 2 documents
        max_df=0.95,         # Ignore terms that appear in more than 95% of documents
        stop_words='english' # Remove English stop words
    )
    
    # Fit and transform training data
    print("Fitting TF-IDF vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform test data
    print("Transforming test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Initialize and train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,  # Increase max iterations for convergence
        C=1.0,          # Regularization parameter
        multi_class='ovr'  # One-vs-Rest for multi-class classification
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer, X_test, y_test


def save_model_and_vectorizer(model, vectorizer, models_dir: str = "models"):
    """
    Save the trained model and vectorizer to disk.
    
    Args:
        model: Trained Logistic Regression model
        vectorizer: Fitted TF-IDF vectorizer
        models_dir: Directory to save the models
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(models_dir, "hs_model.pkl")
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Save the vectorizer
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    print("Model and vectorizer saved successfully!")


def main():
    """
    Main function to orchestrate the training process.
    """
    print("Starting HSCode prediction model training...")
    
    # Define file paths
    data_file = "data/kaggle_data.csv"
    models_dir = "models"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(data_file)
        
        # Train the model
        model, vectorizer, X_test, y_test = train_model(X, y)
        
        # Save the trained model and vectorizer
        save_model_and_vectorizer(model, vectorizer, models_dir)
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {models_dir}/hs_model.pkl")
        print(f"Vectorizer saved to: {models_dir}/tfidf_vectorizer.pkl")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()

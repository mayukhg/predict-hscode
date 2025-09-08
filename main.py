"""
FastAPI application for HSCode prediction.
This module provides a REST API endpoint to predict HSCodes for item descriptions
using a pre-trained machine learning model.
"""

import os
import joblib
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils import preprocess_text


# Initialize FastAPI app
app = FastAPI(
    title="HSCode Prediction API",
    description="API for predicting HSCodes from item descriptions using machine learning",
    version="1.0.0"
)

# Global variables to store loaded model and vectorizer
model = None
vectorizer = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    item_description: str = Field(
        ..., 
        description="Text description of the item to classify",
        min_length=1,
        max_length=1000
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    hscode: str = Field(..., description="Predicted HS code")
    confidence: float = Field(..., description="Confidence score of the prediction (0-1)")


def load_model_and_vectorizer():
    """
    Load the pre-trained model and vectorizer from disk.
    This function is called during application startup.
    """
    global model, vectorizer
    
    try:
        # Define model file paths
        model_path = "models/hs_model.pkl"
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        
        # Check if model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        # Load the model and vectorizer
        print("Loading pre-trained model and vectorizer...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print("Model and vectorizer loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model and vectorizer when the application starts."""
    load_model_and_vectorizer()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HSCode Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_hscode(request: PredictionRequest) -> PredictionResponse:
    """
    Predict HSCode for a given item description.
    
    Args:
        request: PredictionRequest containing item_description
        
    Returns:
        PredictionResponse with predicted HSCode and confidence score
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model or vectorizer not loaded. Please check server logs."
        )
    
    try:
        # Validate input
        if not isinstance(request.item_description, str):
            raise HTTPException(
                status_code=400,
                detail="item_description must be a string"
            )
        
        if len(request.item_description.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="item_description cannot be empty"
            )
        
        # Preprocess the input text
        cleaned_text = preprocess_text(request.item_description)
        
        if len(cleaned_text) == 0:
            raise HTTPException(
                status_code=400,
                detail="Item description contains no valid text after preprocessing"
            )
        
        # Transform text using the fitted vectorizer
        text_vector = vectorizer.transform([cleaned_text])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)
        
        # Get the predicted class (HSCode)
        predicted_class = model.predict(text_vector)[0]
        
        # Get the confidence score (maximum probability)
        confidence = float(max(probabilities[0]))
        
        # Return the prediction
        return PredictionResponse(
            hscode=predicted_class,
            confidence=confidence
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return 500
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

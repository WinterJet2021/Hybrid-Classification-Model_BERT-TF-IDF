# direct_hybrid_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Union
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define InterestClassifier class here (or import it if available)
class InterestClassifier:
    """
    Hybrid Interest Classification model that combines TF-IDF with BERT zero-shot classification
    This is a simplified version for compatibility with the API
    """
    def __init__(self, model_path=None, alpha=0.6, threshold=0.5):
        self.alpha = alpha
        self.threshold = threshold
        self.tfidf_pipeline = None
        self.mlb = None
        self.bert_classifier = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path):
        """Load a saved model from disk"""
        try:
            with open(path, 'rb') as f:
                components = pickle.load(f)
                
            self.tfidf_pipeline = components.get('tfidf_pipeline')
            self.mlb = components.get('mlb')
            self.alpha = components.get('alpha', 0.6)
            self.threshold = components.get('threshold', 0.5)
            
            logger.info(f"Model components loaded from {path}")
            logger.info(f"Model components: {list(components.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, texts, alpha=None, threshold=None, return_scores=False):
        """Predict method adapted for the API"""
        if not isinstance(texts, list):
            texts = [texts]
        
        # Use instance values if not provided
        alpha = alpha if alpha is not None else self.alpha
        threshold = threshold if threshold is not None else self.threshold
        
        if self.tfidf_pipeline is None:
            raise ValueError("TF-IDF pipeline not loaded. Cannot make predictions.")
        
        # Get predictions from TF-IDF pipeline
        text = texts[0]  # Just use the first text for simplicity
        
        # Get raw prediction probabilities
        y_proba = self.tfidf_pipeline.predict_proba([text])
        
        # Convert to dictionary of label -> score
        scores = {}
        for i, label in enumerate(self.mlb.classes_):
            # For MultiOutputClassifier, each element of y_proba is a list of arrays
            # Each array is for one label and has 2 values: [prob_for_0, prob_for_1]
            scores[label] = y_proba[i][0][1]  # Get probability of positive class
        
        # Apply threshold to get labels
        labels = [label for label, score in scores.items() if score >= threshold]
        
        if return_scores:
            # Sort scores for easier interpretation
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'labels': labels,
                'scores': scores,
                'sorted_scores': sorted_scores,
                'alpha': alpha,
                'threshold': threshold
            }
        
        return labels

# Load the hybrid classifier
MODEL_PATH = "hybrid_interest_classifier.pkl"
hybrid_classifier = None

try:
    logger.info(f"Loading hybrid model from {MODEL_PATH}")
    # Create an instance of our classifier and load the model
    hybrid_classifier = InterestClassifier(model_path=MODEL_PATH)
    logger.info("Hybrid model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load hybrid model: {e}")

# Define keyword-based interest detection as fallback
def keyword_interests(text):
    """
    Determine interests using keyword matching as a fallback
    """
    text = text.lower()
    interests = []
    
    if any(word in text for word in ['music', 'band', 'concert', 'sing', 'guitar', 'song']):
        interests.append('Music')
    
    if any(word in text for word in ['food', 'cook', 'recipe', 'restaurant', 'eat', 'cuisine']):
        interests.append('Food')
    
    if any(word in text for word in ['sport', 'gym', 'fitness', 'exercise', 'workout', 'run']):
        interests.append('Sports')
    
    if any(word in text for word in ['art', 'paint', 'draw', 'gallery', 'museum', 'exhibition']):
        interests.append('Arts')
    
    if any(word in text for word in ['tech', 'code', 'software', 'computer', 'programming']):
        interests.append('Technology')
    
    if any(word in text for word in ['learn', 'study', 'course', 'book', 'read', 'class']):
        interests.append('Education')
    
    if any(word in text for word in ['travel', 'trip', 'journey', 'explore', 'hike', 'tourism']):
        interests.append('Travel')
    
    if not interests:
        interests.append('No specific interests detected')
    
    return interests

# Pydantic models
class PredictionRequest(BaseModel):
    text: str
    alpha: Optional[float] = None
    threshold: Optional[float] = None
    return_scores: Optional[bool] = False

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {
        "status": "online", 
        "message": "Hybrid Interest Classifier API is running",
        "model_loaded": hybrid_classifier is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": hybrid_classifier is not None}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict interests based on text input
    """
    text = request.text
    alpha = request.alpha
    threshold = request.threshold
    return_scores = request.return_scores
    
    logger.info(f"Prediction request: text='{text[:50]}...', alpha={alpha}, threshold={threshold}, return_scores={return_scores}")
    
    if not text or text.strip() == "":
        return {"labels": ["No text provided"], "text": text}
    
    if hybrid_classifier is None:
        logger.warning("Using fallback keyword matching (model not loaded)")
        return {"labels": keyword_interests(text), "text": text}
    
    try:
        # Prepare prediction parameters
        kwargs = {}
        if alpha is not None:
            kwargs['alpha'] = alpha
        if threshold is not None:
            kwargs['threshold'] = threshold
        if return_scores:
            kwargs['return_scores'] = True
        
        # Log the call we're about to make
        logger.info(f"Calling hybrid_classifier.predict([{text[:20]}...], {kwargs})")
        
        # Make prediction
        prediction = None
        try:
            # Call predict with the text and kwargs
            prediction = hybrid_classifier.predict([text], **kwargs)
        except TypeError as e:
            # If that fails, try without optional parameters
            logger.warning(f"TypeError with kwargs: {e}. Trying without kwargs.")
            prediction = hybrid_classifier.predict([text])
        
        logger.info(f"Raw prediction: {prediction}")
        
        # Process the prediction result
        labels = []
        scores = {}
        
        # Handle dictionary return type (likely with return_scores=True)
        if isinstance(prediction, dict):
            if 'labels' in prediction:
                labels = prediction['labels']
            
            if return_scores and 'sorted_scores' in prediction:
                scores = dict(prediction['sorted_scores'])
            elif return_scores and 'scores' in prediction:
                scores = prediction['scores']
        
        # Handle list return type
        elif isinstance(prediction, list):
            labels = prediction
        
        # If we still have no labels, use keyword matching
        if not labels:
            logger.warning("No labels detected, using fallback")
            labels = keyword_interests(text)
        
        # Construct response
        response = {"labels": labels, "text": text}
        if return_scores and scores:
            response["scores"] = scores
        
        logger.info(f"Final response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return {"labels": keyword_interests(text), "text": text, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("direct_hybrid_api:app", host="0.0.0.0", port=8000, reload=True)
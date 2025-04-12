import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch
import logging
import time
from typing import List, Dict, Tuple, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define interest categories
INTEREST_CATEGORIES = ["Music", "Food", "Sports", "Technology", "Arts", "Travel", "Education"]

class InterestClassifier:
    """
    Hybrid Interest Classification model that combines TF-IDF with BERT zero-shot classification
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 alpha: float = 0.6, 
                 threshold: float = 0.5,
                 bert_model_name: str = 'facebook/bart-large-mnli',
                 use_gpu: bool = torch.cuda.is_available()):
        """
        Initialize the hybrid classifier
        
        Args:
            model_path: Path to a saved model (if None, a new model will be created)
            alpha: Weight for TF-IDF model (1-alpha for BERT)
            threshold: Classification threshold for final predictions
            bert_model_name: Name of the BERT model to use
            use_gpu: Whether to use GPU for BERT inference
        """
        self.alpha = alpha
        self.threshold = threshold
        self.bert_model_name = bert_model_name
        self.use_gpu = use_gpu
        
        # Initialize models as None
        self.tfidf_pipeline = None
        self.mlb = None
        self.bert_classifier = None
        
        # Load the model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize BERT model
        self._init_bert_classifier()
    
    def _improved_preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing that better preserves domain-specific indicators
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Handle potential NaN values
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters while preserving important separators
        text = re.sub(r'[^\w\s|-]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Define domain terms dictionary
        domain_terms = {
            'music': ['music', 'guitar', 'band', 'concert', 'gig', 'sing', 'song', 'play music', 'musician'],
            'food': ['food', 'cook', 'cuisine', 'recipe', 'restaurant', 'eat', 'culinary', 'bake', 'chef'],
            'sports': ['sport', 'run', 'gym', 'fitness', 'workout', 'exercise', 'athletic', 'training'],
            'arts': ['art', 'paint', 'draw', 'museum', 'gallery', 'exhibit', 'creative', 'design'],
            'technology': ['tech', 'code', 'program', 'software', 'developer', 'computer', 'app', 'digital'],
            'education': ['education', 'learn', 'course', 'class', 'study', 'book', 'read', 'academic'],
            'travel': ['travel', 'trip', 'hike', 'explore', 'tour', 'visit', 'journey', 'destination']
        }
        
        # Check for domain terms and emphasize them
        modified_text = text
        for category, terms in domain_terms.items():
            for term in terms:
                if term in text:
                    # Add the category name explicitly if a related term is found
                    modified_text += f" {category} {category} {term} {term}"
        
        # Split on common separators but preserve the important phrases
        parts = []
        for part in re.split(r'\s*\|\s*', modified_text):
            # Remove numbers (but keep words with numbers like "web3")
            part = re.sub(r'\b\d+\b', '', part)
            parts.append(part)
        
        # Define a more focused stopwords list
        core_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'the', 'a', 'an', 'and', 'but', 
                          'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                          'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                          'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                          'under', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were'}
        
        # Process each part and filter stopwords
        processed_parts = []
        for part in parts:
            words = part.split()
            filtered_words = [word for word in words if word not in core_stopwords]
            
            if filtered_words:
                processed_parts.append(' '.join(filtered_words))
        
        # Join the processed parts back
        processed_text = ' '.join(processed_parts)
        
        return processed_text.strip()
    
    def _init_bert_classifier(self):
        """Initialize the BERT zero-shot classifier"""
        try:
            logger.info(f"Initializing BERT zero-shot classifier with model: {self.bert_model_name}")
            device = 0 if self.use_gpu and torch.cuda.is_available() else -1
            self.bert_classifier = pipeline('zero-shot-classification', 
                                           model=self.bert_model_name, 
                                           device=device)
            logger.info("BERT classifier successfully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BERT classifier: {e}")
            logger.warning("Proceeding without BERT - will use TF-IDF only")
            self.bert_classifier = None
    
    def train(self, 
              df: pd.DataFrame, 
              text_column: str = 'survey_answer', 
              labels_column: str = 'labels_list',
              test_size: float = 0.2):
        """
        Train the TF-IDF + Logistic Regression model
        
        Args:
            df: DataFrame containing survey responses and labels
            text_column: Column name containing the survey responses
            labels_column: Column name containing the labels
            test_size: Proportion of data to use for testing
        
        Returns:
            Evaluation metrics on test set
        """
        logger.info("Starting model training...")
        
        # Prepare labels
        if isinstance(df[labels_column].iloc[0], str):
            logger.info("Converting labels from string to list...")
            # Convert string representation of lists to actual lists
            df[labels_column] = df[labels_column].str.strip('[]').str.split(',')
            # Clean up any extra quotes or spaces
            df[labels_column] = df[labels_column].apply(lambda x: [item.strip().strip("'\"") for item in x])
        
        # Preprocess text
        logger.info("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self._improved_preprocess_text)
        
        # Initialize MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=INTEREST_CATEGORIES)
        y = self.mlb.fit_transform(df[labels_column])
        logger.info(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], y, test_size=test_size, random_state=42, shuffle=True
        )
        logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Create TF-IDF pipeline
        logger.info("Creating and training TF-IDF pipeline...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 3),
            sublinear_tf=True
        )
        
        lr_clf = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            penalty='l2'
        )
        
        multi_lr = MultiOutputClassifier(lr_clf)
        
        self.tfidf_pipeline = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('classifier', multi_lr)
        ])
        
        # Train the pipeline
        self.tfidf_pipeline.fit(X_train, y_train)
        logger.info("TF-IDF pipeline trained successfully")
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred = self.tfidf_pipeline.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
        h_loss = hamming_loss(y_test, y_pred)
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"Hamming Loss: {h_loss:.4f}")
        logger.info(f"Micro F1 Score: {micro_f1:.4f}")
        logger.info(f"Macro F1 Score: {macro_f1:.4f}")
        
        return {
            'hamming_loss': h_loss,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1
        }
    
    def get_tfidf_predictions(self, text: str) -> Dict[str, float]:
        """
        Get predictions from TF-IDF model with confidence scores
        
        Args:
            text: The input text to classify
            
        Returns:
            Dictionary of label -> score
        """
        if self.tfidf_pipeline is None:
            raise ValueError("TF-IDF model is not trained yet. Call train() first.")
            
        # Preprocess text
        processed_text = self._improved_preprocess_text(text)
        
        # Get raw prediction probabilities
        y_proba = self.tfidf_pipeline.predict_proba([processed_text])
        
        # Convert to dictionary of label -> score
        scores = {}
        for i, label in enumerate(self.mlb.classes_):
            # For MultiOutputClassifier, each element of y_proba is a list of arrays
            # Each array is for one label and has 2 values: [prob_for_0, prob_for_1]
            scores[label] = y_proba[i][0][1]  # Get probability of positive class
        
        return scores
    
    def get_bert_predictions(self, text: str) -> Dict[str, float]:
        """
        Get predictions from BERT model
        
        Args:
            text: The input text to classify
            
        Returns:
            Dictionary of label -> score
        """
        if self.bert_classifier is None:
            logger.warning("BERT classifier is not available, returning empty scores")
            return {label: 0.0 for label in INTEREST_CATEGORIES}
            
        try:
            # Use the BERT zero-shot classifier
            result = self.bert_classifier(text, INTEREST_CATEGORIES, multi_label=True)
            
            # Convert to dictionary of label -> score
            scores = dict(zip(result['labels'], result['scores']))
            
            # Ensure all categories are present (BERT may return in different order)
            for category in INTEREST_CATEGORIES:
                if category not in scores:
                    scores[category] = 0.0
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error in BERT prediction: {e}")
            return {label: 0.0 for label in INTEREST_CATEGORIES}
    
    def predict(self, 
                text: str, 
                alpha: Optional[float] = None,
                threshold: Optional[float] = None,
                return_scores: bool = False) -> Union[List[str], Dict]:
        """
        Combine TF-IDF and BERT predictions using weighted average
        
        Args:
            text: The input text to classify
            alpha: Weight for TF-IDF predictions (1-alpha for BERT), uses self.alpha if None
            threshold: Threshold for classification, uses self.threshold if None
            return_scores: Whether to return scores along with labels
        
        Returns:
            Either a list of predicted labels or a dictionary with labels and scores
        """
        if self.tfidf_pipeline is None:
            raise ValueError("Model is not trained yet. Call train() first.")
            
        # Use instance values if not provided
        alpha = alpha if alpha is not None else self.alpha
        threshold = threshold if threshold is not None else self.threshold
        
        # Time the predictions
        start_time = time.time()
        
        # Get TF-IDF predictions
        tfidf_scores = self.get_tfidf_predictions(text)
        tfidf_time = time.time() - start_time
        
        # Get BERT predictions if available
        bert_time_start = time.time()
        if self.bert_classifier is not None:
            bert_scores = self.get_bert_predictions(text)
            use_bert = True
        else:
            bert_scores = {category: 0.0 for category in INTEREST_CATEGORIES}
            use_bert = False
            logger.warning("BERT classifier not available, using TF-IDF only")
        bert_time = time.time() - bert_time_start
        
        # Combine predictions
        combined_scores = {}
        final_labels = []
        
        for category in INTEREST_CATEGORIES:
            # Get scores from both models
            tfidf_score = tfidf_scores.get(category, 0.0)
            bert_score = bert_scores.get(category, 0.0)
            
            # Weighted average (if using BERT)
            if use_bert:
                final_score = (alpha * tfidf_score) + ((1 - alpha) * bert_score)
            else:
                final_score = tfidf_score
                
            combined_scores[category] = final_score
            
            # Apply threshold
            if final_score >= threshold:
                final_labels.append(category)
        
        total_time = time.time() - start_time
        
        if return_scores:
            # Sort scores for easier interpretation
            sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'labels': final_labels,
                'scores': combined_scores,
                'sorted_scores': sorted_scores,
                'tfidf_scores': tfidf_scores,
                'bert_scores': bert_scores,
                'timing': {
                    'tfidf': tfidf_time,
                    'bert': bert_time,
                    'total': total_time
                },
                'alpha': alpha,
                'threshold': threshold,
                'using_bert': use_bert
            }
        
        return final_labels
    
    def save_model(self, path: str = "hybrid_interest_classifier.pkl"):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        if self.tfidf_pipeline is None:
            raise ValueError("Model is not trained yet. Call train() first.")
            
        # Note: We only save the TF-IDF pipeline and MLBinarizer
        # BERT will be re-initialized on load
        components = {
            'tfidf_pipeline': self.tfidf_pipeline,
            'mlb': self.mlb,
            'alpha': self.alpha,
            'threshold': self.threshold,
            'bert_model_name': self.bert_model_name,
            'interest_categories': INTEREST_CATEGORIES,
            'version': '1.0'
        }
        
        with open(path, 'wb') as f:
            pickle.dump(components, f)
            
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a saved model from disk
        
        Args:
            path: Path to the saved model
        """
        try:
            with open(path, 'rb') as f:
                components = pickle.load(f)
                
            self.tfidf_pipeline = components['tfidf_pipeline']
            self.mlb = components['mlb']
            self.alpha = components.get('alpha', 0.6)
            self.threshold = components.get('threshold', 0.5)
            self.bert_model_name = components.get('bert_model_name', 'facebook/bart-large-mnli')
            
            logger.info(f"Model loaded from {path}")
            
            # Re-initialize BERT classifier
            self._init_bert_classifier()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# Example usage
def main():
    try:
        # Load dataset
        logger.info("Loading dataset: survey_interest_dataset_enhanced.csv")
        df = pd.read_csv('survey_interest_dataset_enhanced.csv')
        
        # Convert labels_list if it's a string representation
        if 'labels_list' in df.columns and isinstance(df['labels_list'].iloc[0], str):
            logger.info("Converting labels_list from string to list...")
            df['labels_list'] = df['labels_list'].str.strip('[]').str.split(',')
            df['labels_list'] = df['labels_list'].apply(lambda x: [item.strip().strip("'\"") for item in x])
        
        # Initialize classifier
        logger.info("Initializing classifier with alpha=0.6, threshold=0.5")
        classifier = InterestClassifier(alpha=0.6, threshold=0.5)
        
        # Train the model
        logger.info("Training the model...")
        metrics = classifier.train(df)
        logger.info(f"Training metrics: {metrics}")
        
        # Save the model
        model_path = "hybrid_interest_classifier.pkl"
        logger.info(f"Saving model to {model_path}")
        classifier.save_model(model_path)
        
        # Test on some examples
        test_examples = [
            "I love hiking in the mountains and trying local foods wherever I travel.",
            "I'm a software developer who plays guitar in a band on weekends.",
            "I spend most of my time reading books and attending online courses.",
            "I enjoy painting landscapes and visiting art museums when I travel."
        ]
        
        logger.info("Testing model on example inputs...")
        for example in test_examples:
            result = classifier.predict(example, return_scores=True)
            logger.info(f"\nExample: '{example}'")
            logger.info(f"Predicted interests: {result['labels']}")
            logger.info("Top interests by score:")
            for category, score in result['sorted_scores'][:3]:
                logger.info(f"  {category}: {score:.4f}")
                
        # Fine-tuning alpha parameter demo
        logger.info("\nFine-tuning alpha parameter:")
        example = "I work as a software developer and enjoy hiking on weekends"
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            result = classifier.predict(example, alpha=alpha, return_scores=True)
            logger.info(f"\nAlpha = {alpha} (TF-IDF weight: {alpha}, BERT weight: {1-alpha})")
            logger.info(f"Predicted interests: {result['labels']}")
            logger.info("Top 3 scores:")
            for category, score in result['sorted_scores'][:3]:
                logger.info(f"  {category}: {score:.4f}")
        
        logger.info("Model training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
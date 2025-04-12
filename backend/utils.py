import re
import pandas as pd

def preprocess_text(text):
    """
    Enhanced text preprocessing that better preserves domain-specific indicators
    """
    # Handle potential NaN values
    if text is None or isinstance(text, float) and pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters while preserving important separators
    text = re.sub(r'[^\w\s|-]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Explicitly preserve key domain terms by adding them multiple times
    # This increases their weight in the vectorization
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
    
    # Define a more focused stopwords list (smaller to keep more domain indicators)
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
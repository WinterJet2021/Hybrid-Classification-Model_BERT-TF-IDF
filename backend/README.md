# Hybrid Interest Classifier API

This Hugging Face Space hosts a FastAPI-based machine learning API that predicts user interests (Music, Food, Travel, etc.) from free-text input. It uses a hybrid model combining TF-IDF + BERT zero-shot classification.

Try it by sending a POST request to `/predict` with:
```json
{
  "text": "I love hiking and coding!",
  "alpha": 0.6,
  "threshold": 0.5,
  "return_scores": true
}

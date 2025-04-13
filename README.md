---
title: NomadSync: Hybrid Interest Classifier API
emoji: 🧭
colorFrom: emerald
colorTo: blue
sdk: docker
pinned: true
---

# 🧭 NomadSync: Hybrid Interest Classifier API

This is the official **interest classification microservice** for [NomadSync](https://nomadsync.ai) — a digital platform designed to help **digital nomads** connect with events, people, and opportunities in Thailand.

This API receives free-text responses from users (e.g., “I love art galleries and surfing meetups”) and predicts **interest categories** like `Arts`, `Sports`, `Travel`, or `Technology` using a hybrid machine learning model.

---

## ✨ What It Does

- 🔍 Understands natural language descriptions from users
- 🧠 Combines **TF-IDF + multi-label classifiers** (Logistic Regression, SVM, Random Forest)
- 🛟 Includes **keyword-based fallback** to improve coverage
- 📤 Offers a clean **RESTful API with FastAPI**
- 🌐 Integrated with the NomadSync WordPress frontend via `fetch()`

---

## 💼 Use Case: Personalized Matching in NomadSync

When a new digital nomad signs up on NomadSync, they are asked:

> *“Describe yourself in a few words. What are you into?”*

This API analyzes their response and instantly recommends:
- Relevant events
- Volunteer opportunities
- Potential friend groups
- Business partnerships

It enables personalized, meaningful connections from day one.

---

## 🔁 API Endpoints

### `POST /predict`

#### 🔹 Request
```json
{
  "text": "I love working in cafes, going to indie concerts, and hiking.",
  "return_scores": true
}
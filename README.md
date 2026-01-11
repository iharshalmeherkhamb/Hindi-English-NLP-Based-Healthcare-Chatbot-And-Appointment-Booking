# 🏥 Healthcare Chatbot (AI-Powered Medical Assistant)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![NLP](https://img.shields.io/badge/AI-NLP-green)

## 📜 Overview
**Healthcare Chatbot** is an intelligent, multilingual triage system designed to bridge the gap between patients and medical specialists. Unlike traditional rule-based chatbots, this system uses **Retrieval-Augmented Generation (RAG)** and **Semantic Vector Search** to understand patient symptoms in both **English and Hindi**.

It predicts potential diseases based on symptoms and automatically routes the patient to the correct specialist (e.g., Heart Pain → Cardiologist), simulating a complete hospital appointment booking experience.

## ✨ Key Features
* **🗣️ Multilingual Support:** Accepts inputs in **Hindi** or **English** (Integrated with Google Neural Machine Translation).
* **🧠 Semantic Search:** Uses **HuggingFace Embeddings (`all-MiniLM-L6-v2`)** to understand the *meaning* of symptoms, not just keywords.
* **🎯 Hybrid Input System:** Users can type symptoms or select multiple options from a dynamic dropdown to ensure accuracy.
* **👨‍⚕️ Intelligent Doctor Routing:** Automatically maps diseases to the correct specialist (Cardiologist, Dermatologist, Neurologist, etc.).
* **📅 Appointment Booking:** A simulated booking system with a database of 30+ doctors, fees, and time slots.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Language:** Python
* **NLP & AI:** LangChain, Sentence-Transformers, Scikit-Learn (KNN)
* **Translation:** Deep-Translator (Google API)
* **Data Processing:** Pandas

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Healthcare-Chatbot.git](https://github.com/your-username/Healthcare-Chatbot.git)
cd Healthcare-Chatbot

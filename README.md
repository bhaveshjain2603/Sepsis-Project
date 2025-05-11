# 🩸 Sepsis Prediction & Subtype Classification

This project predicts sepsis and classifies it into medical subtypes using machine learning (Random Forest, KMeans) and a Streamlit web app interface.

## 🔍 Features

- Sepsis classification 
- Subtype prediction using KMeans clustering
- Streamlit-based UI
- AI-generated treatment advice using Gemini API

## 📁 Project Structure

- `data/` - CSV files used for training and testing
- `models/` - Serialized ML models (`.pkl`)
- `src/Streamlit_App/` - Web app (Streamlit)
- `dev/` - Notebook for training and EDA

## 🚀 Running the App

### 1. Clone the Repository

```bash
git clone https://github.com/bhaveshjain2603/Sepsis-Project
cd Streamlit_App
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
streamlit run app.py
```

## ⚙️ Tech Stack
- Frontend: Streamlit (Python-based interactive UI)
- Backend: FastAPI (for RESTful predictions)
- ML Models: Random Forest, Logistic Regression, XGBoost, SVM, Decision Tree, KMeans Clustering
- AI Integration: Gemini API for treatment recommendation
- Data Handling & Processing: Pandas, Scikit-learn, Imbalanced-learn

## 🔮 Core Features
- ✅ Predicts if a patient is at risk of developing sepsis
- 🧬 Classifies the subtype of sepsis using KMeans clustering
- 💊 Suggests AI-powered treatment plans using Gemini or ChatGPT
- 📊 Visualizes feature importance for model explainability
- ⚡ REST API support for model inference via FastAPI

## 📢 Future Enhancements
- 🚨 Real-time alert system for hospital dashboards
- 📱 Mobile-friendly responsive design
- 📁 Patient history tracking and prediction timelines
- 🧠 More advanced subtype classification with medical datasets

## 🤝 Contributing
Contributions are welcome! Feel free to:

- 🌟 Star this repository
- 🐛 Report issues
- 📬 Suggest enhancements
- 🔁 Submit pull requests with improvements or ideas
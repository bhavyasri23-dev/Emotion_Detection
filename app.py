import nltk
nltk.download('stopwords')
nltk.download('punkt')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nest_asyncio
import nltk
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Fix asyncio loop issues in Streamlit
nest_asyncio.apply()


stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

# Load and preprocess data
@st.cache_data
def load_and_train():
    df = pd.read_csv('tweet_emotions.csv')
    df['clean_text'] = df['content'].apply(preprocess)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    return model, vectorizer, df, acc, prec, report

model, vectorizer, df, acc, prec, report = load_and_train()

# Load transformer model
@st.cache_resource
def load_emotion_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

emotion_classifier = load_emotion_classifier()

def predict_emotion(text):
    try:
        result = emotion_classifier(text)
        if isinstance(result, list) and isinstance(result[0], list) and 'label' in result[0][0]:
            return result[0][0]['label']
        else:
            return "neutral"
    except Exception as e:
        return "neutral"

# Streamlit UI
st.title("🧠 Emotion Detection Web App")

st.markdown("Enter a sentence to predict its **emotion** using a machine learning model and a transformer model.")

user_input = st.text_input("Type your message here:")

if st.button("Predict"):
    if user_input:
        cleaned_input = preprocess(user_input)
        vector_input = vectorizer.transform([cleaned_input])
        ml_pred = model.predict(vector_input)[0]
        transformer_pred = predict_emotion(user_input)
        
        st.success(f"🔍 **ML Prediction:** {ml_pred}")
        st.success(f"🤗 **Transformer Prediction:** {transformer_pred}")
    else:
        st.warning("Please enter some text.")

# Metrics Display
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {acc:.2f}")
st.write(f"**Precision (macro):** {prec:.2f}")
st.text("Classification Report:")
st.code(report)

# Visualization
st.subheader("🔎 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x='sentiment', data=df, palette='viridis', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)




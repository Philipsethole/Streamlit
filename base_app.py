"""
    Simple Streamlit webserver application for serving developed classification
    models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal Streamlit web
    application. You are expected to extend the functionality of this script
    as part of your project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Load Vectorizer
news_vectorizer = open("tfidfvect.pkl", "rb")
test_cv = joblib.load(news_vectorizer)  # Load your vectorizer from the pickle file

# Load your raw data
raw = pd.read_csv("train.csv")

# ---- Custom CSS for Styling ----
st.markdown("""
    <style>
        /* App Background */
        .stApp {
            background-color: #f0f2f6;
        }

        /* Title and Subheader Styling */
        h1 {
            color: #1f77b4;
            text-align: center;
        }

        h2, h3 {
            color: #ff5733;
        }

        /* Sidebar Background */
        .css-1d391kg {
            background-color: #2b2b2b !important;
        }

        /* Info Box */
        .stAlert {
            background-color: #ffcccb;
            color: black;
            font-weight: bold;
        }

        /* Button Styling */
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
        }
        
        .stButton > button:hover {
            background-color: #ff5733;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Main App Function ----
def main():
    """News Classifier App with Streamlit"""

    # App Title
    st.title("📰 News Classifier")
    st.subheader("📡 Analyzing and Classifying News Articles")

    # Sidebar Navigation
    options = ["Team Information", "Project Overview", "EDA", "Prediction"]
    selection = st.sidebar.selectbox("📌 Choose Option", options)

    # ---- Team Information Page ----
    if selection == "Team Information":
        st.markdown("<h2 style='color:#ff5733;'>👥 Meet the Team</h2>", unsafe_allow_html=True)
        st.markdown("""
        **Team Members:**
        - Phillip Sethole
        - Musa Khuzwayo 
        - Mpho Onthatile Moloi 
        - Lebo Letsoalo 
        - Kwanda Shandu 
        """)

    # ---- Project Overview Page ----
    elif selection == "Project Overview":
        st.markdown("<h2 style='color:#1f77b4;'>📌 Project Overview</h2>", unsafe_allow_html=True)
        st.markdown("""
        This project focuses on building a classification model to categorize news articles into specific categories.

        **Key Features:**
        - Utilization of three machine learning models.
        - Analyzing and visualizing the data.
        - User-friendly interface for predictions.
        """)

    # ---- Exploratory Data Analysis (EDA) Page ----
    elif selection == "EDA":
        st.markdown("<h2 style='color:#ff5733;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 10px;">
                Visualizations and analysis of the dataset will be displayed here.
            </div>
        """, unsafe_allow_html=True)

        # Display EDA Images
        st.image("EDA1.png", caption="📌 Word Cloud for Most Frequent Words", use_container_width=True)
        st.image("EDA2.png", caption="📌 Top 10 Most Common Words in Each Category", use_container_width=True)

    # ---- Prediction Page ----
    elif selection == "Prediction":
        st.markdown("<h2 style='color:#1f77b4;'>🧠 Prediction with ML Models</h2>", unsafe_allow_html=True)

        # Input Text Box
        st.markdown("<p style='color:#ff5733;'>📜 Enter the news article text for classification:</p>", unsafe_allow_html=True)
        news_text = st.text_area(" ", "Type Here")

        # Model Selection
        st.markdown("<p style='color:#1f77b4;'>📌 Choose a model:</p>", unsafe_allow_html=True)
        model_choice = st.selectbox(" ", ["Logistic Regression", "Support Vector Machine", "Neural Network Classifier"])

        # Predict Button
        if st.button("Classify 🔍"):
            vect_text = test_cv.transform([news_text]).toarray()

            try:
                # Load Model Based on Selection
                if model_choice == "Logistic Regression":
                    predictor = joblib.load(open(os.path.join("best_log_reg_model.pkl"), "rb"))
                elif model_choice == "Support Vector Machine":
                    predictor = joblib.load(open(os.path.join("best_svm_model.pkl"), "rb"))
                elif model_choice == "Neural Network Classifier":
                    predictor = joblib.load(open(os.path.join("best_nn_model.pkl"), "rb"))

                # Make Predictions
                prediction = predictor.predict(vect_text)

                # Display Prediction in a Styled Box
                st.markdown(f"""
                    <div style="background-color:#d4edda; color:#155724; padding:10px; border-radius:10px;">
                        <strong>📝 Text Categorized as: {prediction[0]}</strong>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading model: {e}")

# Run the App
if __name__ == '__main__':
    main()

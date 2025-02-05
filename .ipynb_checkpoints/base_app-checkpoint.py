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

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("tfidfvect.pkl", "rb")
test_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit """

    # Creates a main title and subheader on your page - these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")

    # Creating sidebar with selection box - you can create multiple pages this way
    options = ["Team Information", "Project Overview", "EDA", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Team Information Page
    if selection == "Team Information":
        st.info("Meet the Team")
        st.markdown("""
        **Team Members:**
        - Phillip Sethole
        - Musa Khuzwayo 
        - Mpho Onthatile Moloi 
        - Lebo Letsoalo 
        - Kwanda Shandu 
        """)

    # Project Overview Page
    elif selection == "Project Overview":
        st.info("Project Overview")
        st.markdown("""
        This project focuses on building a classification model to categorize news articles into specific categories.

        Key Features:
        - Utilization of three machine learning models.
        - Analyzing and visualizing the data.
        - User-friendly interface for predictions.
        """)

    # EDA Page
    elif selection == "EDA":
        st.info("Exploratory Data Analysis")
        st.markdown("""
        Visualizations and analysis of the dataset will be displayed here.
        """)
        # Display the images inside the EDA section
        st.image("EDA1.png", caption="EDA Visualization 1", use_column_width=True)
        st.image("EDA2.png", caption="EDA Visualization 2", use_column_width=True)

    # Prediction Page
    elif selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        # Dropdown to select model
        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Support vector machine", "Neural network classifier"])

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = test_cv.transform([news_text]).toarray()

            # Load the selected model
            try:
                if model_choice == "Logistic Regression":
                    predictor = joblib.load(open(os.path.join("best_log_reg_model.pkl"), "rb"))
                elif model_choice == "Support vector machine":
                    predictor = joblib.load(open(os.path.join("best_svm_model.pkl"), "rb"))
                elif model_choice == "Neural network classifier":
                    predictor = joblib.load(open(os.path.join("best_nn_model.pkl"), "rb"))
                # Make predictions
                prediction = predictor.predict(vect_text)

                # Display prediction
                st.success("Text Categorized as: {}".format(prediction))
            except Exception as e:
                st.error(f"Error loading model: {e}")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

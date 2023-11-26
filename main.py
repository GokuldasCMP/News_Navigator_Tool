import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import google.generativeai as palm
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Define categories before using them
categories = ["Business", "Entertainment", "Politics", "Sports", "Tech"]

# Define a function to load the tokenizer with caching
@st.cache_data
def load_tokenizer():
    return AutoTokenizer.from_pretrained("GokulMundott/bert-fine-tuned-tokenize")

# Define a function to load the model with caching
@st.cache_resource
def load_model():
    return TFAutoModelForSequenceClassification.from_pretrained('GokulMundott/bert-fine-tuned-news_cat')

# Streamlit app
st.title("News Navigator 🧭")
user_input = st.text_area("If you have multiple articles separate them with the pipe symbol (|):")
classify_button = st.button("Classify")

# Move this block inside the if block
if classify_button and user_input:
    # Load tokenizer and model
    tokenizer = load_tokenizer()
    model = load_model()

    # Split the input into a list of articles
    articles = [article.strip() for article in user_input.split('|')]

    # Process each article
    for idx, article in enumerate(articles, start=1):  # Start index at 1
        # Tokenize and predict category
        inputs = tokenizer(article,
                           truncation=True,
                           padding=True,
                           return_tensors="tf")
        outputs = model(inputs["input_ids"])[0]
        predicted_category = tf.argmax(outputs, axis=1).numpy()[0]

        # Ensure predicted_category is within the valid range of categories
        predicted_category = min(predicted_category, len(categories) - 1)

        # Map numerical category to label (adjust as needed)
        predicted_label = categories[predicted_category]

        # Display the result with st.info
        st.info(f"The predicted category for Article {idx} is: {predicted_label}")

        # Use the article in the prompt for summary generation
        palm.configure(api_key=API_KEY)
        model_id = 'models/text-bison-001'

        prompt = f'''I will give you a news article. Study the news article and give a summary of it within 100 words.\n{article}'''

        completion = palm.generate_text(
            model=model_id,
            prompt=f"{article}\n{prompt}",
            temperature=0.0,
            max_output_tokens=500,
            candidate_count=1)

        summary = completion.result

        # Display the summary on the Streamlit app
        st.subheader(f"Generated Summary for Article {idx}")
        st.info(summary)

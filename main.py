import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tf_keras as tf
import google.generativeai as palm
from dotenv import load_dotenv
import os
import spacy
import time

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Define categories before using them
categories = ["Business", "Entertainment", "Politics", "Sports", "Tech"]

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Define a function to load the tokenizer with caching
@st.cache_data
def load_tokenizer():
    return AutoTokenizer.from_pretrained("GokulMundott/bert-fine-tuned-tokenize")

# Define a function to load the model with caching
@st.cache_resource
def load_model():
    return TFAutoModelForSequenceClassification.from_pretrained('GokulMundott/bert-fine-tuned-news_cat')


st.title("News Navigator 🧭")

attention_message = st.empty()  # Create an empty space for the attention message

# Animate the attention message
for i in range(5):  # You can adjust the number of frames as needed
    attention_message.text(f"👈 Please go to the sidebar for instructions 👈")
    time.sleep(0.75)  # Adjust the sleep duration for the desired speed

# Clear the animated attention message
attention_message.empty()



# Add instructions and project details to the sidebar
with st.sidebar:
    
    st.markdown("<h2 style='color: #3366ff;'>Welcome to News Navigator 🧭</h2>", unsafe_allow_html=True)

    st.write("News Navigator is a tool designed to analyze and classify news articles. "
             "It utilizes state-of-the-art natural language processing models to predict categories, "
             "perform sentiment analysis, named entity recognition, and generate concise summaries.")

    st.markdown("<h3 style='color: #3366ff;'>How to Use:</h3>", unsafe_allow_html=True)
    st.write("1. Enter one or more articles in the text area below.")
    st.write("2. Separate multiple articles with the pipe symbol (|).")
    st.write("3. Click the 'Classify' button to get predictions, summaries, sentiment analysis, and named entity recognition.")

    st.markdown("<h3 style='color: #3366ff;'>Named Entity Recognition (NER):</h3>", unsafe_allow_html=True)
    st.write("Named Entity Recognition identifies entities such as locations, persons, and organizations in the text. "
             "These entities provide additional context about the content of the news articles.")

    st.markdown("<h3 style='color: #3366ff;'>Project Details:</h3>", unsafe_allow_html=True)
    st.write("This project leverages advanced language models and technologies, including:")
    st.markdown("- **DistillBERT-fine-tuned Tokenizer:** Tokenizes input articles for analysis.")
    st.markdown("- **DistillBERT-fine-tuned News Classification Model:** Predicts categories (Business, Entertainment, Politics, Sports, Tech).")
    st.markdown("- **spaCy NLP Model:** Performs Named Entity Recognition (NER) on entities like locations, persons, and organizations.")
    st.markdown("- **TextBison-001 Model (PaLM 2):** Analyzes sentiment in the context of each news article.")
    st.markdown("- **TextBison-001 Model (PaLM 2):** Generates concise summaries of the given news articles.")

    # Note about the categories with HTML styling
    st.markdown("<div style='color: #3366ff; margin-top: 20px; font-style: italic;'>"
                "<strong>Note:</strong> News Navigator can categorize articles into the following genres: Business, Entertainment, Politics, Sports, and Tech. "
                "Feel free to explore the insights provided for each article."
                "</div>", unsafe_allow_html=True)

    # Beta version note with HTML styling
    st.markdown("<div style='color: #ff9933; margin-top: 20px; font-style: italic;'>"
                "<strong>Beta Version:</strong> This is a beta version of News Navigator. Your feedback is valuable for improvements."
                "</div>", unsafe_allow_html=True)
    
    # Add contact information at the end
    st.markdown("---")
    st.subheader("Contact Information:")
    st.write("For any inquiries or feedback, feel free to reach out:")
    st.write("📧 Email: [gokuldas127199544@gmail.com](mailto:gokuldas127199544@gmail.com)")
    st.write("📷 Instagram: [gokul_mundott](https://www.instagram.com/gokul_mundott/)")




user_input = st.text_area("If you have multiple articles separate them with the pipe symbol (|):")

classify_button = st.button("Classify")


# Move this block inside the if block
if classify_button and user_input:
    # Load tokenizer and model
    tokenizer = load_tokenizer()
    model = load_model()

    # Split the input into a list of articles
    articles = [article.strip() for article in user_input.split('|')]

    # Check if the input articles are valid
    valid_articles = [article for article in articles if len(article.split()) > 10]  # Adjust the threshold as needed

    if not valid_articles:
        st.warning("Please enter valid articles. Nonsensical input or very short articles detected.")
    else:
        # Process each article
        for idx, article in enumerate(valid_articles, start=1):  # Start index at 1
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

            # Use spaCy for named entity recognition
            doc = nlp(article)  # Create a new spaCy NLP object for each article
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Filter entities to show only 'GPE', 'PERSON', and 'ORG'
            filtered_entities = [(ent, label) for ent, label in entities if label in ['GPE', 'PERSON', 'ORG']]

            # Display named entity recognition result as a table
            st.subheader(f"Named Entity Recognition for Article {idx}")
            st.table(filtered_entities)

            # Sentiment Analysis Prompt
            sentiment_prompt = f'''Consider the context of the news article and analyze its sentiment:\n{article}'''

            # Use the article in the prompt for summary generation
            palm.configure(api_key=API_KEY)
            sentiment_model_id = 'models/text-bison-001'

            sentiment_completion = palm.generate_text(
                model=sentiment_model_id,
                prompt=sentiment_prompt,
                temperature=0.0,
                max_output_tokens=500,
                candidate_count=1)

            sentiment_result = sentiment_completion.result

            # Display the sentiment on the Streamlit app
            st.subheader(f"Sentiment Analysis for Article {idx}")

            # Color-coded sentiment display
            if isinstance(sentiment_result, str) and "positive" in sentiment_result.lower():
                st.success(sentiment_result)
            elif isinstance(sentiment_result, str) and "negative" in sentiment_result.lower():
                st.error(sentiment_result)
            else:
                st.info(sentiment_result)


            # Summary Generation Prompt
            summary_prompt = f'''I will give you a news article. Study the news article and give a summary of it within 100 words.\n{article}'''
            model_id = 'models/text-bison-001'

            completion = palm.generate_text(
                model=model_id,
                prompt=summary_prompt,
                temperature=0.0,
                max_output_tokens=500,
                candidate_count=1)

            summary = completion.result

            # Display the summary on the Streamlit app
            st.subheader(f"Generated Summary for Article {idx}")

            # Styled box for summary display with theme-aware background color
            background_color = st.get_option("theme.backgroundColor")
            border_color = "#3366ff"  # Adjust as needed

            st.markdown(
                f"<div style='border: 1px solid {border_color}; border-radius: 10px; padding: 10px; background-color: {background_color};'>{summary}</div>",
                unsafe_allow_html=True
            )

import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import google.generativeai as genai
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

            # Named Entity Recognition Prompt
            ner_prompt = f'''Identify entities in the following news article, including locations (GPE), persons, and organizations. 
            Return results as a list of (entity, type) tuples:\n{article}'''

            # Use the gemini-1.5-flash model to generate the NER results
            genai.configure(api_key=API_KEY)
            ner_model=genai.GenerativeModel("gemini-1.5-flash")


            # Use the gemini-1.5-flash model to generate the NER results
            ner_response = ner_model.generate_content(ner_prompt)
            ner_result = ner_response.text

            # Process and display the NER result if it matches the expected output format
            st.subheader(f"Named Entity Recognition for Article {idx}")

            # Attempt to parse the result if it's in a structured format, otherwise display as is
            try:
                # Assuming the output is structured as text with tuples, parse it
                # For example, it might look like "(Suresh Raina, PERSON), (New Zealand, GPE)"
                parsed_entities = eval(ner_result)  # Evaluate string into a list of tuples if it's safe and structured
                if isinstance(parsed_entities, list) and all(isinstance(item, tuple) for item in parsed_entities):
                    # Creating a formatted list of entities
                    st.markdown("### Extracted Entities:")
                    for entity, entity_type in parsed_entities:
                        # Style each entity differently based on its type
                        if entity_type == 'PERSON':
                            st.markdown(f"🔹 **{entity}** (Person)")
                        elif entity_type == 'GPE':
                            st.markdown(f"🔸 **{entity}** (Location)")
                        elif entity_type == 'ORGANIZATION':
                            st.markdown(f"🔶 **{entity}** (Organization)")
                        else:
                            st.markdown(f"• **{entity}** ({entity_type})")
                else:
                    # Directly show the raw output if parsing fails
                    st.markdown(f"**Raw Output:** {ner_result}")
            except Exception as e:
                # Display raw output if parsing fails
                st.markdown(f"**Raw Output:** {ner_result}")


            # Sentiment Analysis Prompt
            sentiment_prompt = f'''Consider the context of the news article and analyze its sentiment:\n{article}'''

            # Use the article in the prompt for sentiment generation
            sentiment_model = genai.GenerativeModel("gemini-1.5-flash")

            sentiment_response = sentiment_model.generate_content(sentiment_prompt)
            sentiment_result = sentiment_response.text

            # Display the sentiment on the Streamlit app
            st.subheader(f"Sentiment Analysis for Article {idx}")

            # Color-coded sentiment display with custom colors
            if isinstance(sentiment_result, str):
                sentiment_result = sentiment_result.lower()
                
                if "positive" in sentiment_result:
                    # Display positive sentiment with a green background
                    st.markdown(f"<div style='background-color: #28a745; color: white; padding: 10px; border-radius: 5px;'>{sentiment_result}</div>", unsafe_allow_html=True)
                
                elif "negative" in sentiment_result:
                    # Display negative sentiment with a red background
                    st.markdown(f"<div style='background-color: #dc3545; color: white; padding: 10px; border-radius: 5px;'>{sentiment_result}</div>", unsafe_allow_html=True)
                
                else:
                    # Display neutral sentiment with a gray background
                    st.markdown(f"<div style='background-color: #6c757d; color: white; padding: 10px; border-radius: 5px;'>{sentiment_result}</div>", unsafe_allow_html=True)



            # Summary Generation Prompt
            summary_prompt = f'''I will give you a news article. Study the news article and give a summary of it within 100 words.\n{article}'''
            summary_response = sentiment_model.generate_content(summary_prompt)
            summary = summary_response.text

            # Display the summary on the Streamlit app
            st.subheader(f"Generated Summary for Article {idx}")

            # Styled box for summary display with theme-aware background color
            background_color = st.get_option("theme.backgroundColor")
            border_color = "#3366ff"  # Adjust as needed

            st.markdown(
                f"<div style='border: 1px solid {border_color}; border-radius: 10px; padding: 10px; background-color: {background_color};'>{summary}</div>",
                unsafe_allow_html=True
            )
            
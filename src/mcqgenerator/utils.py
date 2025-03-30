import os
import sys
import json
import pandas as pd
import traceback
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.abspath("src"))
from src.mcqgenerator.utils import get_table_data, read_file
from src.mcqgenerator.MCQGenerator import generate_chain

load_dotenv()

# Strict JSON prompt template
JSON_PROMPT_TEMPLATE = """
You are a quiz generator that MUST output in this exact JSON format:
{
  "1": {
    "mcq": "question text",
    "options": {
      "a": "option 1",
      "b": "option 2",
      "c": "option 3",
      "d": "option 4"
    },
    "correct": "a"
  }
}

Generate {number} MCQs about {subject} ({tone} level) from this text:
{text}

RULES:
1. Output MUST be valid JSON only
2. No other text or commentary
3. All options must be distinct
4. Only one correct answer per question
"""

def load_response_json():
    """Load the response template"""
    path = os.path.join("data", "Response.json")
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        st.error("Response.json not found")
        st.stop()
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        st.stop()

def extract_json_from_response(response):
    """Robust JSON extractor from model response"""
    content = response.content if hasattr(response, 'content') else str(response)
    
    # Case 1: Pure JSON
    if content.strip().startswith('{'):
        try:
            return json.loads(content), None
        except json.JSONDecodeError:
            pass
    
    # Case 2: Markdown JSON
    if '```json' in content:
        try:
            json_part = content.split('```json')[1].split('```')[0].strip()
            return json.loads(json_part), None
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Case 3: No JSON found
    return None, content

# Main App
def main():
    st.title("MCQ Generator")
    
    with st.form("input_form"):
        upload_file = st.file_uploader("Upload file", type=["txt", "pdf"])
        mcq_count = st.number_input("Number of MCQs", 3, 50, 10)
        subject = st.text_input("Subject", max_chars=50)
        tone = st.text_input("Difficulty", "Medium", max_chars=50)
        
        if st.form_submit_button("Generate"):
            if not all([upload_file, subject, tone]):
                st.error("Please fill all fields")
            else:
                generate_quiz(upload_file, mcq_count, subject, tone)

def generate_quiz(file, count, subject, tone):
    """Handle the quiz generation process"""
    with st.spinner("Generating..."):
        try:
            text = read_file(file)
            if not text:
                st.error("Failed to read file")
                return
            
            response_json = load_response_json()
            
            # Generate with strict JSON instructions
            response = generate_chain.invoke({
                "text": text,
                "number": count,
                "subject": subject,
                "tone": tone,
                "response_json": json.dumps(response_json),
                "instructions": JSON_PROMPT_TEMPLATE
            })
            
            quiz_data, feedback = extract_json_from_response(response)
            
            if quiz_data:
                show_quiz_table(quiz_data)
            else:
                handle_failed_generation(feedback)
                
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            traceback.print_exc()

def show_quiz_table(quiz_data):
    """Display the generated quiz"""
    table_data = get_table_data(quiz_data)
    if table_data:
        df = pd.DataFrame(table_data).set_index("Question")
        st.table(df)
    else:
        st.error("Failed to process quiz data")

def handle_failed_generation(feedback):
    """Handle cases when JSON generation fails"""
    st.error("Failed to generate valid quiz format")
    if feedback:
        st.subheader("Model Feedback")
        st.write(feedback)
    st.info("""
    Tips to fix:
    1. Try with more detailed input text
    2. Reduce the number of questions
    3. Check the subject is clear
    4. Try a different difficulty level
    """)

if __name__ == "__main__":
    main()
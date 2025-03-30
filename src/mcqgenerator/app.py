import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import traceback
import logging
from langchain_ai21 import ChatAI21
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configure logging and environment
load_dotenv()
logging.basicConfig(level=logging.INFO)

class FileReader:
    @staticmethod
    def read_file(file):
        """Read content from various file types with error handling"""
        try:
            if file.name.endswith('.pdf'):
                return FileReader._read_full_pdf(file)
            elif file.name.endswith('.docx'):
                return FileReader._read_full_docx(file)
            else:  # Assume text file
                return FileReader._read_full_text(file)
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    @staticmethod
    def _read_full_pdf(file):
        """Extract all text from PDF including all pages"""
        reader = PyPDF2.PdfReader(file)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:  # Only add non-empty pages
                full_text.append(text)
        return "\n".join(full_text)
    
    @staticmethod
    def _read_full_docx(file):
        """Extract all text from Word document"""
        doc = Document(file)
        return "\n".join(para.text for para in doc.paragraphs if para.text)

    @staticmethod
    def _read_full_text(file):
        """Read complete text file content"""
        return file.getvalue().decode("utf-8")
class MCQGeneratorApp:
    def __init__(self):
        self.config = self._get_default_config()
        try:
            self.llm = self._init_langchain_ai21()
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    def _init_langchain_ai21(self):
        """Initialize AI21 language model with proper configuration"""
        api_key = os.getenv("AI21_API_KEY")
        if not api_key:
            raise ValueError("AI21_API_KEY not found in environment variables")
        
        try:
            return ChatAI21(
                ai21_api_key=api_key,
                model="jamba-instruct",
                temperature=0.7,
                max_tokens=2000
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI21 model: {str(e)}")

    def _get_default_config(self):
        """Default configuration for quiz generation"""
        return {
            "quiz_format": {
                "options": ["a", "b", "c", "d"],
                "required_fields": ["mcq", "options", "correct"]
            }
        }

    def _preprocess_text(self, text):
        """Clean and prepare text for processing"""
        # Remove excessive whitespace and special characters
        text = ' '.join(text.split())
        return text.encode('ascii', errors='ignore').decode()

    def generate_quiz(self, text, count, difficulty):
        """Generate quiz strictly from provided text content"""
        text = self._preprocess_text(text)
        
        prompt_template = """Generate EXACTLY {count} {difficulty}-difficulty multiple choice questions USING ONLY THIS TEXT:
        
        {text}

        STRICT REQUIREMENTS:
        1. You MUST generate exactly {count} questions - no more, no less
        2. EVERY question MUST be directly answerable from this EXACT text
        3. Format each question EXACTLY like this example:
        {{
            "mcq": "What language is MongoDB written in?",
            "options": {{
            "a": "Python",
            "b": "JavaScript",
            "c": "C++",
            "d": "Java"
            }},
            "correct": "c"
        }}
        4. Number the questions sequentially from 1 to {count} in the "quiz" object
        5. DO NOT include any questions that can't be answered from this text
        6. DO NOT repeat similar questions
        7. Return ONLY the JSON output with NO additional text

        OUTPUT FORMAT:
        {{
        "quiz": {{
            "1": {{...}},
            "2": {{...}},
            ...
            "{count}": {{...}}
        }}
        }}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["count", "difficulty", "text"]
        )

        try:
            chain = (
                {"count": RunnablePassthrough(),
                "difficulty": RunnablePassthrough(),
                "text": RunnablePassthrough()}
                | prompt
                | self.llm
            )

            response = chain.invoke({
                "count": count,
                "difficulty": difficulty,
                "text": text[:10000]  # Increased character limit
            })

            if hasattr(response, 'content'):
                response = response.content

            quiz_data = self._parse_response(response)
            if not quiz_data or len(quiz_data.get("quiz", {})) < count:
                raise ValueError(f"Only got {len(quiz_data.get('quiz', {}))} questions instead of {count}")

            return self._validate_quiz(quiz_data, text)

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            if "quiz_data" in locals():
                st.json(quiz_data)  # Show partial output for debugging
            return None
    def _parse_response(self, response):
        """Parse and validate the AI response"""
        try:
            response = response.strip()
            
            # Extract JSON from markdown if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            return json.loads(response)
        except Exception as e:
            st.warning(f"Failed to parse response: {str(e)}")
            return None

    def _validate_quiz(self, quiz_data, source_text):
        """Ensure questions are derived from source text"""
        source_lower = source_text.lower()
        valid_questions = {}
        
        for q_id, question in quiz_data.get("quiz", {}).items():
            q_text = question["mcq"].lower()
            correct_opt = question["correct"]
            correct_answer = question["options"][correct_opt].lower()
            
            # Verify question or correct answer exists in source
            if (q_text in source_lower or 
                correct_answer in source_lower):
                valid_questions[q_id] = question
        
        if not valid_questions:
            raise ValueError("No valid questions found in source text")
            
        return {"quiz": valid_questions}

    def display_quiz(self, quiz_data):
        """Display the generated quiz in a table"""
        try:
            questions = []
            for q_num, q in quiz_data.get("quiz", {}).items():
                questions.append({
                    "#": q_num,
                    "Question": q["mcq"],
                    "Options": "\n".join(f"{k}) {v}" for k, v in q["options"].items()),
                    "Answer": f"{q['correct']}) {q['options'][q['correct']]}"
                })
            
            st.dataframe(
                pd.DataFrame(questions),
                hide_index=True,
                column_config={
                    "Question": st.column_config.TextColumn(width="wide"),
                    "Options": st.column_config.TextColumn(width="medium")
                }
            )
        except Exception as e:
            st.error(f"Display error: {str(e)}")
            st.json(quiz_data)

    def run(self):
        """Main application interface"""
        st.title("MCQ Generator from Documents")
        
        with st.form("quiz_form"):
            file = st.file_uploader("Upload PDF/DOCX/TXT", 
                                  type=["pdf", "docx", "txt"])
            col1, col2 = st.columns(2)
            with col1:
                count = st.slider("Number of Questions", 1, 50, 5)
            with col2:
                difficulty = st.selectbox("Difficulty", 
                                        ["Easy", "Medium", "Hard"])
            
            if st.form_submit_button("Generate Quiz"):
                if not file:
                    st.error("Please upload a file")
                else:
                    try:
                        text = FileReader.read_file(file)
                        if len(text) < 100:
                            st.error("Document too short (min 100 characters)")
                            return
                            
                        with st.spinner(f"Generating {count} questions..."):
                            quiz = self.generate_quiz(text, count, difficulty)
                            
                            if quiz:
                                st.success("Quiz generated successfully!")
                                self.display_quiz(quiz)
                                with st.expander("View source text"):
                                    st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
                            else:
                                st.error("Failed to generate valid quiz")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    app = MCQGeneratorApp()
    app.run()
import os
from dotenv import load_dotenv
from langchain_ai21 import ChatAI21
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
key = os.environ.get("AI21_API_KEY")
if not key:
    raise ValueError("AI21_API_KEY not found in environment variables.")

# Initialize the LLM
llm = ChatAI21(api_key=key, model="jamba-instruct", temperature=0.5)

# In your MCQGenerator.py or wherever you define your prompt:

PROMPT_TEMPLATE = """
You are an expert MCQ generator. Given a text input, you must generate {number} multiple-choice questions about {subject} with a {tone} tone.
Your output MUST be in JSON format with the following structure:
{{
  "1": {{
    "mcq": "question text",
    "options": {{
      "a": "option 1",
      "b": "option 2",
      "c": "option 3",
      "d": "option 4"
    }},
    "correct": "correct_option_key"
  }},
  "2": {{
    ...
  }}
}}

IMPORTANT RULES:
1. Only output the JSON object, nothing else
2. Don't include any commentary or feedback
3. Ensure all questions are distinct
4. Make sure options are plausible but only one is correct

Text Input:
{text}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone"],
    template=PROMPT_TEMPLATE
)

# Define the review prompt
review_template = """
You are an expert in English grammar and writing. Review the multiple-choice quiz for {subject} students.
Evaluate the complexity of the questions given below and analyze if they match the cognitive and analytical abilities of the students.
If necessary, update any question that needs revision. Keep your response within 50 words.
### Quiz:
{quiz}
"""

quiz_review_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=review_template
)

# Chains
quiz_chain = quiz_generation_prompt | llm
review_chain = quiz_review_prompt | llm

# ✅ Fixed Function: Extract quiz content while keeping the subject
def extract_quiz_output(output, inputs):
    """Extracts generated quiz content while retaining subject. Prevents missing key errors."""
    subject = inputs.get("subject", "Unknown")  # 🔹 Fix: Provide a default subject to prevent KeyError
    return {"quiz": output.content, "subject": subject}

# ✅ Fixed Function: Ensure subject is included before review
def combine_with_subject(inputs):
    """Combine quiz output with subject before review."""
    return {"quiz": inputs.get("quiz", "No quiz generated"), "subject": inputs.get("subject", "Unknown")}

# ✅ Fixed RunnableLambda to avoid missing `inputs`
generate_chain = (
    quiz_generation_prompt
    | llm
    | RunnableLambda(lambda output, **kwargs: extract_quiz_output(output, kwargs.get("inputs", {})))  # 🔹 Fix: Now ensures inputs always exists
    | RunnableLambda(combine_with_subject)  # 🔹 Fix: Ensures subject is passed correctly
    | review_chain
)

import json
import os
import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CREDENTIALS_PATH = './dochq-staging-72db3155a22d.json'
if os.path.exists(CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
else:
    logger.error(f"Credentials file not found at {CREDENTIALS_PATH}")

g_client = genai.Client(vertexai=True, project="dochq-staging", location="us-central1")

def classify_response(user_input: str, expected_options: list[str]) -> str:
    prompt = f"""
        You're a medical assistant. A patient replied: "{user_input}"
        Given the following valid options: {expected_options}, try classifying their response into one of these or 'question'.
        Respond with only the option word.
        """
    response = g_client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=500,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
        )
    )
    return(response.text.strip())

def classify_node(user_input: str, expected_options: list[str]) -> str:
    prompt = f"""
        You're a medical assistant. A patient replied: "{user_input}"
        Given the following valid options: {expected_options}, try classifying their response into one of these.
        Respond with only the option word.
        """
    response = g_client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=500,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
        )
    )
    return(response.text.strip())

def rephrase_question(node_name: str, question: str, options: list[str]) -> str:
    print(f"Rephrasing question: {question}")
    prompt = f"""
        You are a helpful, compassionate virtual medical assistant guiding an Europian user through a conversation.
        You're currently at step '{node_name}' in a flow.
        Your task is to rephrase this question in a friendly, clear way: "{question}"  
        Also, Present the following options in a natural way to guide the user: {options}.
        Keep it concise and avoid medical jargon.

        Return your response as a JSON object with two fields:
        - "response": the rephrased question
        - "options": list of rephrased options

        Example format:
        {{ "response": "...", "options": [...] }}
        """
    response = g_client.models.generate_content(
        model="gemini-1.5-flash-002",
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.5,
            top_p=0.8,
            max_output_tokens=500,
            response_mime_type='application/json',
            # response_schema=CountryInfo,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
        )
    )
    try:
        return json.loads(response.text)
    except Exception as e:
        return {
            "response": question,
            "options": options
        }

def handle_fallback(question: str, options: str, user_message: str):
    print(f"Handling fallback for question: {question} with user message: {user_message}")
    fallback_prompt = f"""
        A user was asked this triage question: "{question}"
        Valid options were: {options}
        Instead, they asked: "{user_message}"

        IF what they asked is related to conversation happening: 
        - please respond with a gentle clarification or mini answer if possible.
        ELSE:
        - Apologize for you are just triage bot and cannot answer their question.
        Then, ask the original question again, encouraging them to choose from options

        Respond in this JSON format:
        {{ "response": "...", "options": [...] }}
        """
    response = g_client.models.generate_content(
        model="gemini-1.5-flash-002",
        contents=[fallback_prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=500,
            response_mime_type='application/json',
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
        )
    )
    try:
        return json.loads(response.text)
    except Exception:
        return {
            "response": question,
            "options": options
        }
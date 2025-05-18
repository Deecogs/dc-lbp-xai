import functions_framework
from google import genai
from google.genai import types
import base64
import json
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure credentials path is set correctly
CREDENTIALS_PATH = './credentials.json'
if os.path.exists(CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
else:
    logger.error(f"Credentials file not found at {CREDENTIALS_PATH}")

def extract_json_with_action(text):
    """
    Extract a JSON object with 'response' and 'action' keys from the text.
    If extraction fails, create a default response.
    """
    try:
        # Try to find a JSON object
        match = re.search(r'\{.*?"response".*?"action".*?\}', text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
            
            # Validate the JSON has required keys
            if 'response' in parsed_json and 'action' in parsed_json:
                return parsed_json
        
        # If no valid JSON found, create a default response
        logger.warning(f"Could not extract valid JSON. Original text: {text}")
        return {
            "response": text,
            "action": "continue"
        }
    
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return {
            "response": "I'm having trouble understanding the response. Could you please repeat?",
            "action": "continue"
        }

def generate(contents):
    try:
        # Initialize client with error handling
        try:
            client = genai.Client(
                vertexai=True,
                project="dochq-staging",
                location="us-central1",
            )
        except Exception as client_init_error:
            logger.error(f"Failed to initialize Genai Client: {client_init_error}")
            raise

        # System prompt (kept as in original code)
        system_prompt = """You are Alia, a physiotherapy AI assistant. You currently just deal with patients having problem or pain in their body. You follow the below tree to diagnose the patient and recommend them with appropriate next steps be it self-care or physiotherapy exercises or book a doctor\\'s appointment. Be polite, empath and crisp with your responses. And Respond in the below format:
{"response": ..., "action": ...}, Make sure your response should always be in required JSON format do not send anything else.
where, "response" is the question which needs to be asked and "action" will depend on the next steps and can take following values: close_chat, continue, next_api, camera_on.
   ####
   TREE YAML:
   - Greet the person and introduce yourself only by your name and ask how can you assist them today.
   - Analyze the patient's pain location. If it is lower back proceed further else tell them that "I'm just capable of handling lower back pains right now, check back after some time".
   - Express your concern about the pain in the lower back area and ask for a quick assessment to better understand the problem.
   - If the user says YES
    - to further confirm on the pain location ask the patient to show where exactly he/she is experiencing pain.
    - Respond by telling the body part name only shown by the user in the video along with "Thank you for showing me the pain location." and action being next_api in above given JSON format.
   - If the user says NO
    - Respond with "You can visit us sometime later so that we can assist you better." and action being close_chat in given JSON format."""

        # Configure model generation
        model = "gemini-1.5-flash-002"
        generate_content_config = types.GenerateContentConfig(
            temperature=0.5,
            top_p=0.8,
            max_output_tokens=1024,
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
            system_instruction=[types.Part.from_text(text=system_prompt)],
        )

        # Generate content with robust error handling
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            return response.text.replace("\n", "")
        except Exception as generation_error:
            logger.error(f"Content generation failed: {generation_error}")
            raise

    except Exception as e:
        logger.error(f"Unexpected error in generate function: {e}")
        raise

@functions_framework.http
def hello_http(request):
    # CORS handling
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    # Standard headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json"
    }

    try:
        # Robust request data extraction
        request_json = request.get_json(silent=True) or {}
        request_args = request.args or {}

        # Extract chat history
        chat_history = request_json.get('chat_history') or request_args.get('chat_history')
        
        if not chat_history:
            error_response = {
                'error': 'No chat history provided',
                'response': "I couldn't process your request.",
                'action': 'close_chat'
            }
            return (json.dumps({
                'success': False, 
                'statusCode': 400, 
                'data': error_response
            }), 400, headers)

        # Process chat history
        contents = []
        for item in chat_history:
            try:
                if 'user' in item:
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=item['user'])]))
                if 'response' in item:
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=item['response'])]))
                if 'video' in item:
                    try:
                        video_data = base64.b64decode(item['video'])
                        contents.append(types.Content(role="user", parts=[types.Part.from_bytes(data=video_data, mime_type="video/mp4")]))
                    except Exception as video_error:
                        logger.error(f"Video processing error: {video_error}")
                        # Continue processing other items even if one video fails
            except Exception as item_error:
                logger.error(f"Error processing chat history item: {item_error}")
                continue

        # Generate response
        try:
            res = generate(contents)
            logger.info(f"Raw response: {res}")

            # Extract JSON with action
            response_data = extract_json_with_action(res)

            final_response = {
                'success': True, 
                'statusCode': 200, 
                'data': response_data
            }
            return (json.dumps(final_response), 200, headers)

        except Exception as generation_error:
            logger.error(f"Response generation error: {generation_error}")
            error_response = {
                'response': "I'm having trouble processing your request right now.",
                'action': 'close_chat'
            }
            final_response = {
                'success': False, 
                'statusCode': 500, 
                'data': error_response
            }
            return (json.dumps(final_response), 500, headers)

    except Exception as e:
        logger.error(f"Unexpected error in hello_http: {e}")
        error_response = {
            'response': "An unexpected error occurred. Please try again later.",
            'action': 'close_chat'
        }
        final_response = {
            'success': False, 
            'statusCode': 500, 
            'data': error_response
        }
        return (json.dumps(final_response), 500, headers)



# import functions_framework
# import langchain_google_vertexai
# from langchain_google_vertexai import ChatVertexAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnableLambda
# from langchain_core.output_parsers import JsonOutputParser
# from pydantic import BaseModel
# from langchain_core.messages import HumanMessage, AIMessage
# import base64
# import json
# import os
# import logging
# import re

# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Set up credentials
# CREDENTIALS_PATH = './credentials.json'
# if os.path.exists(CREDENTIALS_PATH):
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
# else:
#     logger.error(f"Credentials file not found at {CREDENTIALS_PATH}")

# # System Prompt
# SYSTEM_PROMPT = """You are Alia, a physiotherapy AI assistant dealing with patients having pain in their body.
# Follow this YAML tree for diagnosis and recommendation. Keep responses empathetic and in this JSON format only:
# {{"response": "...", "action": "..."}}
# Valid "action" values: close_chat, continue, next_api, camera_on.

# TREE:
# - Greet and ask how you can help.
# - If pain is in lower back, proceed. Else say: "I'm just capable of handling lower back pains right now."
# - Ask for quick assessment.
# - If YES:
#     - Ask to show the pain location on video.
#     - Respond with identified body part and say "Thank you for showing..." with action next_api.
# - If NO:
#     - Say: "You can visit us sometime later..." with action close_chat.
# """

# # LangChain model setup
# llm = ChatVertexAI(
#     model_name="gemini-1.5-flash-002",
#     temperature=0.5,
#     project="dochq-staging",
#     location="us-central1"
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{user_input}")
# ])

# class AIResponse(BaseModel):
#     response: str
#     action: str
# # Output parser to extract the correct JSON keys
# parser = JsonOutputParser(pydantic_object=AIResponse)

# # Final chain combining prompt and LLM
# chain = prompt | llm | parser


# @functions_framework.http
# def hello_http(request):
#     # CORS headers
#     if request.method == "OPTIONS":
#         headers = {
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "GET,POST",
#             "Access-Control-Allow-Headers": "Content-Type",
#             "Access-Control-Max-Age": "3600",
#         }
#         return ("", 204, headers)

#     headers = {
#         "Access-Control-Allow-Origin": "*",
#         "Content-Type": "application/json"
#     }

#     try:
#         request_json = request.get_json(silent=True) or {}
#         chat_history = request_json.get("chat_history", [])

#         if not chat_history:
#             return (json.dumps({
#                 "success": False,
#                 "statusCode": 400,
#                 "data": {
#                     "response": "No chat history provided.",
#                     "action": "close_chat"
#                 }
#             }), 400, headers)

#         # Convert chat history to LangChain format
#         lc_chat_history = []
#         for item in chat_history:
#             if "user" in item:
#                 lc_chat_history.append(HumanMessage(content=item["user"]))
#             elif "response" in item:
#                 lc_chat_history.append(AIMessage(content=item["response"]))
#             elif "video" in item:
#                 logger.warning("Video input detected, skipping for now in LangChain integration.")

#         # Add dummy latest user input for triggering generation
#         user_input = chat_history[-1].get("user", "How can I assist you today?")

#         # Run LangChain pipeline
#         try:
#             result = chain.invoke({"chat_history": lc_chat_history, "user_input": user_input})
#             return (json.dumps({
#                 "success": True,
#                 "statusCode": 200,
#                 "data": result
#             }), 200, headers)
#         except Exception as gen_err:
#             logger.error(f"LangChain pipeline error: {gen_err}")
#             return (json.dumps({
#                 "success": False,
#                 "statusCode": 500,
#                 "data": {
#                     "response": "I'm having trouble processing your request right now.",
#                     "action": "close_chat"
#                 }
#             }), 500, headers)

#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         return (json.dumps({
#             "success": False,
#             "statusCode": 500,
#             "data": {
#                 "response": "An unexpected error occurred. Please try again later.",
#                 "action": "close_chat"
#             }
#         }), 500, headers)
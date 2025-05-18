# import functions_framework
# from google import genai
# from google.genai import types
# import base64
# import json
# import os
# import traceback
# import logging
# import sys
# from datetime import datetime

# # Configure logging
# log_format = '%(asctime)s - %(levelname)s - %(message)s'
# logging.basicConfig(
#     level=logging.INFO,
#     format=log_format,
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # Create a function to log and print at the same time for maximum visibility
# def log_and_print(level, message):
#     """Log and print a message for maximum visibility"""
#     print(f"[{datetime.now().isoformat()}] {level}: {message}")
#     if level == "INFO":
#         logger.info(message)
#     elif level == "WARNING":
#         logger.warning(message)
#     elif level == "ERROR":
#         logger.error(message)
#     elif level == "DEBUG":
#         logger.debug(message)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials.json'

# def safe_json_loads(json_str, default=None):
#     """
#     Safely parse JSON with extensive error handling.
#     Returns default value if parsing fails.
#     """
#     if not json_str:
#         return default
        
#     # Try to determine if this is a JSON string
#     stripped = json_str.strip()
#     if not (stripped.startswith('{') or stripped.startswith('[')):
#         return default
        
#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError as e:
#         log_and_print("ERROR", f"JSON parse error: {str(e)} in: {json_str[:100]}...")
        
#         # Try to recover by doing some basic cleanup
#         try:
#             # Replace single quotes with double quotes
#             fixed = json_str.replace("'", '"')
#             # Handle trailing commas in objects and arrays
#             fixed = fixed.replace(",}", "}")
#             fixed = fixed.replace(",]", "]")
#             return json.loads(fixed)
#         except:
#             log_and_print("ERROR", f"Couldn't recover JSON after cleanup attempt")
#             return default

# def generate(contents):
#     """
#     Generate a response using Google's generative AI model.
    
#     Args:
#         contents: The chat history contents to send to the model
        
#     Returns:
#         The model's generated text response
        
#     Raises:
#         Exception: If any error occurs during the API call
#     """
#     try:
#         log_and_print("INFO", f"Starting generate function with {len(contents)} items in history")
        
#         # Log each message in the contents for debugging
#         for i, content in enumerate(contents):
#             log_and_print("DEBUG", f"Content {i} role: {content.role}")
#             if content.parts and len(content.parts) > 0:
#                 log_and_print("DEBUG", f"Content {i} text: {content.parts[0].text[:100]}...")

#         client = genai.Client(
#             vertexai=True,
#             project="dochq-staging",
#             location="us-central1"
#         )

#         system_prompt = """You are Alia, a physiotherapy AI assistant. You currently just deal with patients having problem or pain in their lower back. 
#   You follow the below tree to diagnose the patient and recommend them with appropriate next steps be it self-care or physiotherapy exercises or book a doctor's appointment. 
#   Be polite, empath and crisp with your responses. Give options for the answers along with your response.
#   All responses should strictly be in the JSON format as below:
#     {{'question': response, 'options': [..., ..., ...]}, \"action\": ...}
# where, \"response\" is the question which needs to be asked, \"options\" comes respectively with the question and \"action\" will depend on the next steps and can take following values: continue, rom_api, dashboard_api, close_chat.

#   IMPORTANT: Never repeat questions that have already been asked and answered. Always check previous questions and answers before deciding what to ask next. If you notice questions being repeated, move to the next logical step in the tree.
  
#   The user might ask some clarifying questions in between, answer them and continue back with the appropriate tree path. 
#   The user might change the answer to some previous question which he/she answered incorrectly before, analyze it and continue back from the the appropriate tree path question if the correction changes the tree path else acknowledge the user about the pain and continue the same path.
#   Reply with JSON only and nothing else.

# TREE:
# - You first enquire about the intensity of the lower back pain of the patient with 0 being no pain and 10 being the most severe pain.
# - Then ask the user whether bending forward or leaning backward is causing more discomfort.
# - Follow by enquiring whether the pain started after an accident, injury, or a sudden strain on your back?
#   - Classify the user's response under one of the categories - SPONTANEOUS or TRAUMA. If SPONTANEOUS, check whether the user is also having leg pain?
#     - If Yes, follow up by enquiring whether the user is having pain in one leg or both the legs?
#       - If in Both Legs, Mark it as a critical situation and ask the user to address symptoms like numbness around perianal area, difficulty with bladder control or controlled urination, or loss of full rectal sensation.
#         - If any or all are positive, ask the user to immediately report to ED and CLOSE the chat.
#         - If all are negative, ask the user to address symptoms like fever, history of cancer, drug use, HIV, osteoporosis or night-time pain.
#           - If any or all are positive, ask the user to take pain reliefs and if the situation persists for more than 2-3 days, advise them to consult a doctor because the pain can possibly be due to some infection, tumor or lumbar compression fracture and CLOSE the chat.
#           - If all are negative, ask the user to take video assessment for further understanding of the cause.
#             - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
#             - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
#       - If in One Leg, ask the user to address symptoms like fever, history of cancer, drug use, HIV, osteoporosis or night-time pain.
#         - If any or all are positive, ask the user to take pain reliefs and if the situation persists for more than 2-3 days, advise them to consult a doctor because the pain can possibly be due to some infection, tumor or lumbar compression fracture and CLOSE the chat.
#         - If all are negative, ask the user to take video assessment for further understanding of the cause.
#           - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
#           - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
#     - If No, request the user to tell you whether it's getting worse as the days go by.
#       - If Yes, ask the user to take video assessment for further understanding of the cause.
#         - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
#         - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
#       - If No, ask the user to take video assessment as it can be possibly because of lumbar strain.
#         - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
#         - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
#   - If TRAUMA, and the user has explained the situation well, continue; else, ask the user to explain the instance better.
#     - If the situation can be classified as violence or road accident, Inform the user that reporting such cases is recommended, so advice them to reach out to the appropriate authorities. Else continue.
#     - Next, question the user whether it's harder to walk than usual.
#       - If Yes, ask if any part of their leg like knee, ankle, toe feel weak or unstable while walking.
#         - If Yes, ask the user to immediately report to ED and CLOSE the chat.
#         - If No, ask the user to take video assessment as it can be possibly because of lumbar strain or fracture.
#           - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
#           - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
#       - If No, ask the user to take video assessment as it can be possibly because of stable fracture or no fracture.
#         - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
#         - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\""""

#         model = "gemini-1.5-flash-002"
        
#         log_and_print("INFO", f"Sending contents to model: {json.dumps([{'role': c.role, 'text': c.parts[0].text[:50] + '...' if c.parts else 'No text'} for c in contents], default=str)}")
        
#         generate_content_config = types.GenerateContentConfig(
#             temperature = 0.2,
#             top_p = 0.8,
#             max_output_tokens = 2000,
#             response_modalities = ["TEXT"],
#             safety_settings = [types.SafetySetting(
#                 category="HARM_CATEGORY_HATE_SPEECH",
#                 threshold="OFF"
#             ),types.SafetySetting(
#                 category="HARM_CATEGORY_DANGEROUS_CONTENT",
#                 threshold="OFF"
#             ),types.SafetySetting(
#                 category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                 threshold="OFF"
#             ),types.SafetySetting(
#                 category="HARM_CATEGORY_HARASSMENT",
#                 threshold="OFF"
#             )],
#             system_instruction=[types.Part.from_text(text=system_prompt)],
#         )

#         log_and_print("INFO", "Calling Gemini model...")
#         response = client.models.generate_content(
#             model = model,
#             contents = contents,
#             config = generate_content_config,
#         )
        
#         log_and_print("INFO", f"Received response from model: {response.text[:200]}...")
#         return (response.text.replace("\n", ""))
    
#     except Exception as e:
#         log_and_print("ERROR", f"Error in generate function: {str(e)}")
#         log_and_print("ERROR", traceback.format_exc())
#         raise

# @functions_framework.http
# def xai_qna(request):
#     """
#     HTTP function to handle the physiotherapy QnA service.
    
#     Args:
#         request: The HTTP request object
        
#     Returns:
#         A tuple containing:
#         - The response JSON
#         - HTTP status code
#         - Response headers
#     """
#     # Record start time for performance tracking
#     start_time = datetime.now()
#     request_id = f"{start_time.strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
    
#     log_and_print("INFO", f"Request {request_id} started")
    
#     # Set CORS headers for all responses
#     headers = {
#         "Access-Control-Allow-Origin": "*",
#         "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
#         "Access-Control-Allow-Headers": "Content-Type",
#         "Access-Control-Max-Age": "3600",
#     }
    
#     # Handle preflight OPTIONS request
#     if request.method == "OPTIONS":
#         log_and_print("INFO", f"Request {request_id}: Handling OPTIONS preflight request")
#         return ("", 204, headers)
    
#     try:
#         # Parse request data
#         request_json = request.get_json(silent=True)
#         request_args = request.args
        
#         # Log the raw request for debugging
#         log_and_print("INFO", f"Request {request_id}: Raw request: {request.data.decode('utf-8')[:1000] if hasattr(request, 'data') else 'No data'}")
        
#         if request_json:
#             log_and_print("INFO", f"Request {request_id}: Parsed JSON: {json.dumps(request_json)[:1000]}...")
        
#         # Extract chat history from request
#         chat_history = None
#         if request_json and 'chat_history' in request_json:
#             chat_history = request_json['chat_history']
#             log_and_print("INFO", f"Request {request_id}: Found chat_history in JSON body with {len(chat_history)} items")
#         elif request_args and 'chat_history' in request_args:
#             # Try to parse chat_history from URL args - it would be a string
#             try:
#                 chat_history = json.loads(request_args['chat_history'])
#                 log_and_print("INFO", f"Request {request_id}: Found chat_history in URL args with {len(chat_history)} items")
#             except json.JSONDecodeError:
#                 log_and_print("ERROR", f"Request {request_id}: Failed to parse chat_history from URL args")
#                 chat_history = None
        
#         if not chat_history:
#             log_and_print("ERROR", f"Request {request_id}: Missing or invalid chat_history parameter")
#             error_response = {'error': 'Missing or invalid chat_history parameter'}
#             final_response = {'success': False, 'statusCode': 400, 'data': error_response}
#             return (json.dumps(final_response), 400, headers)
        
#         # Log the first few entries of chat history for debugging
#         for i, entry in enumerate(chat_history[:5]):
#             log_and_print("INFO", f"Request {request_id}: Chat history entry {i}: {json.dumps(entry)}")
        
#         # Format chat history for the model
#         contents = []
#         previous_questions = set()
#         conversation_log = []
        
#         for i, item in enumerate(chat_history):
#             # Debug log
#             log_and_print("DEBUG", f"Processing history item {i}: {json.dumps(item)}")
            
#             # Track message flow for debugging
#             message_type = "unknown"
#             message_content = "none"
            
#             # Handle user messages
#             if 'user' in item:
#                 user_message = item['user']
#                 log_and_print("INFO", f"Request {request_id}: Adding user message: {user_message}")
#                 contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
#                 message_type = "user"
#                 message_content = user_message
            
#             # Handle response messages (primary format)
#             if 'response' in item:
#                 message_type = "response"
#                 message_content = item['response'][:100] + "..." if len(item['response']) > 100 else item['response']
                
#                 log_and_print("DEBUG", f"Processing response message: {item['response'][:200]}...")
                
#                 # Parse the response to extract the question
#                 try:
#                     response_obj = safe_json_loads(item['response'], {})
#                     question_text = None
                    
#                     # Try to extract question from different possible formats
#                     if response_obj and 'question' in response_obj:
#                         question_text = response_obj['question']
#                         log_and_print("INFO", f"Found question in response: {question_text}")
#                     elif response_obj and 'data' in response_obj and isinstance(response_obj['data'], dict) and 'question' in response_obj['data']:
#                         question_text = response_obj['data']['question']
#                         log_and_print("INFO", f"Found question in response.data: {question_text}")
#                     elif isinstance(item['response'], str) and not response_obj:
#                         # If not JSON but a string, try to use it as is
#                         question_text = item['response']
#                         log_and_print("INFO", f"Using string response as question: {question_text[:100]}...")
                    
#                     if question_text:
#                         # Check if this question has been asked before
#                         if question_text not in previous_questions:
#                             log_and_print("INFO", f"Request {request_id}: Adding assistant message: {question_text}")
#                             contents.append(types.Content(role="assistant", parts=[types.Part.from_text(text=question_text)]))
#                             previous_questions.add(question_text)
#                         else:
#                             log_and_print("WARNING", f"Request {request_id}: Skipping duplicate question: {question_text}")
#                     else:
#                         log_and_print("WARNING", f"Request {request_id}: Couldn't extract question from response: {item['response'][:200]}...")
                
#                 except Exception as e:
#                     log_and_print("ERROR", f"Request {request_id}: Error processing response: {str(e)}")
#                     # Don't stop processing if one entry fails
            
#             # For backward compatibility with older format
#             if 'assistant' in item:
#                 message_type = "assistant"
#                 message_content = item['assistant'][:100] + "..." if len(item['assistant']) > 100 else item['assistant']
                
#                 question_text = item['assistant']
#                 if question_text and question_text not in previous_questions:
#                     log_and_print("INFO", f"Request {request_id}: Adding assistant message (from 'assistant' field): {question_text}")
#                     contents.append(types.Content(role="assistant", parts=[types.Part.from_text(text=question_text)]))
#                     previous_questions.add(question_text)
#                 else:
#                     log_and_print("WARNING", f"Request {request_id}: Skipping duplicate or empty assistant message: {question_text}")
            
#             # Add to conversation log for debugging
#             conversation_log.append({
#                 "index": i,
#                 "type": message_type,
#                 "content": message_content
#             })
        
#         # Summarize conversation for debugging
#         log_and_print("INFO", f"Request {request_id}: Conversation summary: {len(contents)} total messages, {len(previous_questions)} unique questions")
#         log_and_print("INFO", f"Request {request_id}: Conversation flow: {json.dumps(conversation_log)}")
        
#         # If we have no valid content, provide a default starter
#         if len(contents) == 0:
#             log_and_print("WARNING", f"Request {request_id}: No valid content found in chat history. Adding default starter.")
#             contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Hello, I have lower back pain.")]))
        
#         # Call the model
#         log_and_print("INFO", f"Request {request_id}: Calling generate function with {len(contents)} messages")
#         res = generate(contents)
        
#         # Process response
#         if isinstance(res, str):
#             log_and_print("INFO", f"Request {request_id}: Raw response from model: {res[:500]}...")
            
#             # Try to extract the JSON from the response if it has the code wrapper
#             try:
#                 cleaned_res = res
#                 if "```json" in res and "```" in res:
#                     # Extract content between ```json and ```
#                     start = res.find("```json") + 7
#                     end = res.rfind("```")
#                     if start > 7 and end > start:  # Valid positions
#                         cleaned_res = res[start:end].strip()
#                         log_and_print("INFO", f"Request {request_id}: Extracted JSON from code block: {cleaned_res[:200]}...")
                
#                 response_data = safe_json_loads(cleaned_res)
                
#                 if not response_data:
#                     log_and_print("WARNING", f"Request {request_id}: Couldn't parse response as JSON, trying to extract JSON object")
#                     # Try to find a JSON object in the string
#                     json_start = cleaned_res.find('{')
#                     json_end = cleaned_res.rfind('}')
#                     if json_start >= 0 and json_end > json_start:
#                         json_str = cleaned_res[json_start:json_end+1]
#                         response_data = safe_json_loads(json_str)
#                         log_and_print("INFO", f"Request {request_id}: Extracted JSON object: {json_str[:200]}...")
                
#                 if not response_data:
#                     # Last resort: create a basic response
#                     log_and_print("WARNING", f"Request {request_id}: Creating default response data")
#                     response_data = {}
                
#                 log_and_print("INFO", f"Request {request_id}: Processed response data: {json.dumps(response_data)}")
                
#                 # Validate response has the expected format
#                 if 'question' not in response_data:
#                     log_and_print("WARNING", f"Request {request_id}: Response missing 'question' field")
#                     response_data['question'] = "I need more information about your lower back pain. Could you provide additional details?"
                
#                 if 'options' not in response_data:
#                     log_and_print("WARNING", f"Request {request_id}: Response missing 'options' field")
#                     response_data['options'] = ["Yes", "No"]
                
#                 if 'action' not in response_data:
#                     log_and_print("WARNING", f"Request {request_id}: Response missing 'action' field")
#                     response_data['action'] = "continue"
                
#                 # Normalize action field to lowercase
#                 if response_data['action'] == "ROM API":
#                     response_data['action'] = "rom_api"
#                 elif response_data['action'] == "DASHBOARD API":
#                     response_data['action'] = "dashboard_api"
                
#                 # Check if the new question is a repeat of a previous one
#                 if response_data['question'] in previous_questions:
#                     log_and_print("WARNING", f"Request {request_id}: Model returned a duplicate question! Providing a recovery response.")
#                     response_data['question'] = "Let me continue with the assessment. Based on what you've told me so far, I'd like to know more about your symptoms. Have you experienced any other symptoms along with your back pain?"
                
#                 # Prepare final response
#                 final_response = {'success': True, 'statusCode': 200, 'data': response_data}
#                 log_and_print("INFO", f"Request {request_id}: Sending successful response: {json.dumps(final_response)}")
                
#                 # Calculate and log request duration
#                 duration = (datetime.now() - start_time).total_seconds()
#                 log_and_print("INFO", f"Request {request_id} completed in {duration:.2f} seconds")
                
#                 return (json.dumps(final_response), 200, headers)
            
#             except Exception as e:
#                 log_and_print("ERROR", f"Request {request_id}: Error processing model response: {str(e)}")
#                 log_and_print("ERROR", traceback.format_exc())
                
#                 # Provide a fallback response
#                 fallback_response = {
#                     'question': "I'm sorry, I'm having trouble processing your information. Could you please tell me more about your lower back pain?",
#                     'options': ["It started recently", "I've had it for a while", "It comes and goes"],
#                     'action': "continue"
#                 }
                
#                 error_response = {'error': 'Error processing model response', 'details': str(e)}
#                 final_response = {'success': True, 'statusCode': 200, 'data': fallback_response}
                
#                 # Calculate and log request duration
#                 duration = (datetime.now() - start_time).total_seconds()
#                 log_and_print("INFO", f"Request {request_id} completed with fallback in {duration:.2f} seconds")
                
#                 return (json.dumps(final_response), 200, headers)
#         else:
#             log_and_print("ERROR", f"Request {request_id}: Unexpected response type from generate function: {type(res)}")
            
#             # Provide a fallback response
#             fallback_response = {
#                 'question': "I'm sorry, I encountered an issue. Could you please describe your lower back pain again?",
#                 'options': ["It's sharp pain", "It's a dull ache", "It's both"],
#                 'action': "continue"
#             }
            
#             final_response = {'success': True, 'statusCode': 200, 'data': fallback_response}
            
#             # Calculate and log request duration
#             duration = (datetime.now() - start_time).total_seconds()
#             log_and_print("INFO", f"Request {request_id} completed with fallback in {duration:.2f} seconds")
            
#             return (json.dumps(final_response), 200, headers)
    
#     except Exception as e:
#         log_and_print("ERROR", f"Request {request_id}: Uncaught exception: {str(e)}")
#         log_and_print("ERROR", traceback.format_exc())
        
#         # Provide a fallback response even for major errors
#         fallback_response = {
#             'question': "I apologize, but I'm experiencing technical difficulties. Could you please describe your symptoms one more time?",
#             'options': ["My back hurts", "I have pain when moving", "I need help with an injury"],
#             'action': "continue"
#         }
        
#         error_response = {'error': 'An error occurred', 'details': str(e)}
#         final_response = {'success': True, 'statusCode': 200, 'data': fallback_response}
        
#         # Calculate and log request duration
#         duration = (datetime.now() - start_time).total_seconds()
#         log_and_print("ERROR", f"Request {request_id} failed in {duration:.2f} seconds")
        
#         return (json.dumps(final_response), 200, headers)





import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import List, Dict

from functions_framework import http
from flask import Request

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.messages import AIMessage, HumanMessage

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def log_and_print(level, message):
    print(f"[{datetime.now().isoformat()}] {level}: {message}")
    getattr(logger, level.lower())(message)

# Ensure credentials are set
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials.json'

# LLM and Prompt Setup
parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(VertexAI(temperature=0), parser)

llm = VertexAI(
    model_name="gemini-1.5-flash-002",
    temperature=0.2,
    top_p=0.8,
    max_output_tokens=2000
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Alia, a physiotherapy AI assistant. You currently just deal with patients having problem or pain in their lower back. 
    You follow the below tree to diagnose the patient and recommend them with appropriate next steps be it self-care or physiotherapy exercises or book a doctor's appointment. 
    Be polite, empath and crisp with your responses. Give options for the answers along with your response.
    Respond in strict JSON only format:
    {{
        "question": "text",
        "options": ["...", "...", "..."],
        "action": "continue|rom_api|dashboard_api|close_chat"
    }}
    NEVER repeat questions, always reference chat history. Be empathetic and concise.
    The user might ask some clarifying questions in between, answer them and continue back with the appropriate tree path. 
    The user might change the answer to some previous question which he/she answered incorrectly before, analyze it and continue back from the the appropriate tree path question if the correction changes the tree path else acknowledge the user about the pain and continue the same path.
    Reply with JSON only and nothing else.

TREE:
- You first enquire about the intensity of the lower back pain of the patient with 0 being no pain and 10 being the most severe pain.
- Then ask the user whether bending forward or leaning backward is causing more discomfort.
- Follow by enquiring whether the pain started spontaneously or after a trauma in an indirect way.
  - Classify the user's response under one of the categories - SPONTANEOUS or TRAUMA. If SPONTANEOUS, check whether the user is also having leg pain?
    - If Yes, follow up by enquiring whether the user is having pain in one leg or both the legs?
      - If in Both Legs, Mark it as a critical situation and ask the user to address symptoms like numbness around perianal area, difficulty with bladder control or controlled urination, or loss of full rectal sensation.
        - If any or all are positive, ask the user to immediately report to ED and CLOSE the chat.
        - If all are negative, ask the user to address symptoms like fever, history of cancer, drug use, HIV, osteoporosis or night-time pain.
          - If any or all are positive, ask the user to take pain reliefs and if the situation persists for more than 2-3 days, advise them to consult a doctor because the pain can possibly be due to some infection, tumor or lumbar compression fracture and CLOSE the chat.
          - If all are negative, ask the user to take video assessment for further understanding of the cause.
            - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
            - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
      - If in One Leg, ask the user to address symptoms like fever, history of cancer, drug use, HIV, osteoporosis or night-time pain.
        - If any or all are positive, ask the user to take pain reliefs and if the situation persists for more than 2-3 days, advise them to consult a doctor because the pain can possibly be due to some infection, tumor or lumbar compression fracture and CLOSE the chat.
        - If all are negative, ask the user to take video assessment for further understanding of the cause.
          - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
          - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
    - If No, request the user to tell you whether it's getting worse as the days go by.
      - If Yes, ask the user to take video assessment for further understanding of the cause.
        - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
        - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
      - If No, ask the user to take video assessment as it can be possibly because of lumbar strain.
        - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
        - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
  - If TRAUMA, and the user has explained the situation well, continue; else, ask the user to explain the instance better.
    - If the situation can be classified as violence or road accident, Inform the user that reporting such cases is recommended, so advice them to reach out to the appropriate authorities. Else continue.
    - Next, question the user whether it's harder to walk than usual.
      - If Yes, ask if any part of their leg like knee, ankle, toe feel weak or unstable while walking.
        - If Yes, ask the user to immediately report to ED and CLOSE the chat.
        - If No, ask the user to take video assessment as it can be possibly because of lumbar strain or fracture.
          - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
          - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
      - If No, ask the user to take video assessment as it can be possibly because of stable fracture or no fracture.
        - If the user agrees for video assessment, respond with \"Let's Start!\" and give an action for \"rom_api\"
        - If the user disagrees for video assessment, respond with \"Sure, I'll analyze your responses only.\" and give an action for \"dashboard_api\"
"""),
    MessagesPlaceholder(variable_name="chat_history"),
])

chain = prompt | llm | fixing_parser

# Function to convert history into LangChain messages
def convert_chat_history(chat_history: List[Dict]) -> List:
    messages = []
    previous_questions = set()

    for item in chat_history:
        if 'user' in item:
            messages.append(HumanMessage(content=item['user']))
        elif 'response' in item:
            try:
                response_obj = json.loads(item['response'])
                question = response_obj.get('question', item['response'])
                if question not in previous_questions:
                    messages.append(AIMessage(content=question))
                    previous_questions.add(question)
            except Exception as e:
                log_and_print("WARNING", f"Could not parse response JSON: {e}")
        elif 'assistant' in item:
            if item['assistant'] not in previous_questions:
                messages.append(AIMessage(content=item['assistant']))
                previous_questions.add(item['assistant'])

    return messages

@http
def xai_qna(request: Request):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "3600",
    }

    if request.method == "OPTIONS":
        return ("", 204, headers)

    try:
        body = request.get_json(silent=True)
        if not body or 'chat_history' not in body:
            return (json.dumps({
                "success": False,
                "statusCode": 400,
                "data": {"error": "Missing 'chat_history'"}
            }), 400, headers)

        chat_history = body['chat_history']
        messages = convert_chat_history(chat_history)

        log_and_print("INFO", f"Sending {len(messages)} messages to LLM")

        result = chain.invoke({"chat_history": messages}, config={'callbacks': [ConsoleCallbackHandler()]})

        log_and_print("INFO", f"Model output: {result}")

        return (json.dumps({
            "success": True,
            "statusCode": 200,
            "data": result
        }), 200, headers)

    except Exception as e:
        log_and_print("ERROR", f"Exception occurred: {e}")
        log_and_print("ERROR", traceback.format_exc())
        return (json.dumps({
            "success": False,
            "statusCode": 500,
            "data": {"error": str(e)}
        }), 500, headers)

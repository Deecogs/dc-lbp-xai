import functions_framework
from google import genai
from google.genai import types
import base64
import json
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials.json'

def generate(contents):
  client = genai.Client(
      vertexai=True,
      project="dochq-staging",
      location="us-central1"
  )

  system_prompt = """You are in conversation with a patient getting examined for the pain. You are given the chat history of some of the assessment questions answered by them to better understand the condition. There was a video assessment also done to analyze the range of motion of the respective join.
Your task is to interpret and analyze the complete assessment of the patient and give the results in the following JSON format:
{'symptoms': [..., ..., ...], 'possible_diagnosis': [..., ..., ...], 'next_steps': ...}
where,
1. 'symptoms' is the list of signs of illness of the patient.
2. 'possible_diagnosis' is a list of potential medical conditions that could be causing the patient's symptoms, based on their medical history, video examination, and initial assessment.
3. next steps are to create a treatment plan and educate the patient about their condition. The patient may also need follow-up tests and consultations to monitor their progress. Categorize the next steps decided in one the following categories: 'Self-care or Medication', 'Physiotherapy', 'ED'. Give only the category as output."""

  model = "gemini-1.5-flash-002"
  contents = contents
  generate_content_config = types.GenerateContentConfig(
    temperature = 0.2,
    top_p = 0.8,
    max_output_tokens = 1024,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    system_instruction=[types.Part.from_text(text=system_prompt)],
  )

  response = client.models.generate_content(
    model = model,
    contents = contents,
    config = generate_content_config,
  )
  return (response.text.replace("\n", ""))

@functions_framework.http
def hello_http(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",  # Allow requests from any origin
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",  # Cache preflight response for 1 hour
        }
        return ("", 204, headers)
    headers = {"Access-Control-Allow-Origin": "*"}
    request_json = request.get_json(silent=True)
    request_args = request.args
    print(request_json)
    if request_json and 'content' in request_json:
        content = request_json['content']
    elif request_args and 'content' in request_args:
        content = request_args['content']
    try:
        res = generate(json.dumps(content))
        print(res[7:-3].strip())
        if isinstance(res, str):
            res_ = json.loads(res[7:-3].strip())
            response = {'response': res_, 'action': 'close_chat'}
            final_response = {'success': True, 'statusCode': 200, 'data': response}
            return (json.dumps(final_response), 200, headers)
            # response = make_response(json.dumps(response_data), 200)
        else:
            error_response = {'error': 'Unexpected response type from generate function'}
            final_response = {'success': False, 'statusCode': 500, 'data': error_response}
            return (json.dumps(final_response), 500, headers)
            # response = make_response(json.dumps(error_response), 500)
    except Exception as e:
        error_response = {'error': 'An error occurred', 'details': str(e)}
        final_response = {'success': False, 'statusCode': 500, 'data': error_response}
        return (json.dumps(final_response), 500, headers)
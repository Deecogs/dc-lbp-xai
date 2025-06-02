from fastapi import FastAPI
from pydantic import BaseModel
import json
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils_xai import classify_response, classify_node, rephrase_question, handle_fallback

app = FastAPI()
logger = logging.getLogger("uvicorn.error")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("xai_tree.json") as f:
    TRIAGE_TREE = json.load(f)

class ChatPayload(BaseModel):
    chat_history: list[dict]

def get_current_node(message: str) -> str:
    options = TRIAGE_TREE.keys()
    classification = classify_node(message, options)
    print(f"Classified message '{message}' as node '{classification}'")
    return classification

@app.post("/chat")
def chat(payload: ChatPayload):
    headers = {"Content-Type": "application/json"}

    def wrap_success(data: dict):
        return JSONResponse(
            status_code=200,
            content={"success": True, "statusCode": 200, "data": data},
            headers=headers
        )

    def wrap_error(error_msg: str = "I'm having trouble processing your request right now."):
        error_data = {"response": error_msg, "action": "close_chat"}
        return JSONResponse(
            status_code=500,
            content={"success": False, "statusCode": 500, "data": error_data},
            headers=headers
        )
    try:
        chat_history = payload.chat_history
        message = chat_history[-1]["user"] if chat_history else ""
        current_node = get_current_node(chat_history)
        node_data = TRIAGE_TREE.get(current_node, {})
        print(f"Current node: {current_node}")

        if node_data.get("end"):
            return wrap_success({"response": node_data["question"], "options": [], "action": "close_chat"})

        options = list(node_data["options"])
        if len(options) >= 1:
            chosen_option = classify_response(message, options)
            print(f"Classified user response '{message}' as option '{chosen_option}'")
            if chosen_option not in options:
                fallback_response = handle_fallback(node_data['question'], node_data["options"], message)
                return wrap_success({**fallback_response, "action": node_data["action"]})
                # rephrased_q = rephrase_question(current_node, node_data["question"], options, chat_history[user_id])
                # return wrap_success({**rephrased_q, "action": node_data["action"]})

        if 'next_node_map' in node_data:
            next_node = node_data["next_node_map"][chosen_option]
        elif 'next_node' in node_data:
            next_node = node_data["next_node"]

        next_node_data = TRIAGE_TREE[next_node]

        if next_node_data.get("end"):
            return wrap_success({"response": next_node_data["question"], "options": [], "action": "close_chat"})
        else:
            rephrased_question = rephrase_question(next_node, next_node_data["question"], list(next_node_data["options"]))
            return wrap_success({**rephrased_question, "action": node_data["action"]})
    
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return wrap_error()
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import json
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils_bpi import classify_response, rephrase_question, handle_fallback, run_gemini_on_video

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("bpi_tree.json") as f:
    TREE = json.load(f)

class ChatPayload(BaseModel):
    chat_history: list[dict]
    video: Optional[str] = None

def get_current_node(message: str) -> str:
    options = TREE.keys()
    classification = classify_response(message, options)
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
        video_data = payload.video
        current_node = get_current_node(chat_history)
        node_data = TREE.get(current_node, {})
        print(f"Current node: {current_node}")

        if current_node == "greeting":
            next_node = node_data["next_node"]
            next_node_data = TREE[next_node]
            rephrased_q = json.loads(rephrase_question(next_node, next_node_data["question"]))
            return wrap_success({"response": rephrased_q["response"], "action": next_node_data["action"]})

        elif current_node == "identify_issue":
            print("starting identify_issue node")
            options = list(node_data["next_node_map"].keys())
            classification = classify_response(message, options)
            if classification not in options:
                fallback = handle_fallback(node_data["question"], message)
                return wrap_success({**fallback, "action": "continue"})
            next_node = node_data["next_node_map"][classification]
            next_node_data = TREE[next_node]
            rephrased_q = json.loads(rephrase_question(next_node, next_node_data["question"]))
            return wrap_success({"response": rephrased_q["response"], "action": next_node_data["action"]})

        elif current_node == "assess_lower_back_issue":
            options = list(node_data["next_node_map"].keys())
            classification = classify_response(message, options)
            if classification not in options:
                fallback = handle_fallback(node_data["question"], message)
                return wrap_success({**fallback, "action": "continue"})
            next_node = node_data["next_node_map"][classification]
            next_node_data = TREE[next_node]
            if next_node_data["action"] == "close_chat":
                return wrap_success({"response": next_node_data["question"], "action": "close_chat"})
            rephrased_q = json.loads(rephrase_question(next_node, next_node_data["question"]))
            return wrap_success({"response": rephrased_q["response"], "action": next_node_data["action"]})

        elif current_node == "see_pain_location":
            if video_data:
                gemini_response = run_gemini_on_video(video_data)
                if gemini_response == "Others":
                    gemini_response = "body pain"
                next_node = node_data["next_node"]
                next_node_data = TREE[next_node]
                rephrased_q = json.loads(rephrase_question(next_node, next_node_data["question"]))
                return wrap_success({"response": f"Understood! Your {gemini_response.lower()} is troubling you."+rephrased_q["response"], "action": next_node_data["action"]})
            else:
                return wrap_success({"response": "Please upload a short video showing where you are experiencing pain.", "action": node_data["action"]})

        elif current_node in ["end_conversation", "other_issue"]:
            return wrap_success({"response": node_data["question"], "action": node_data["action"]})

        else:
            return wrap_success({"response": "Sorry, I’m not sure how to continue. Let's start over.", "action": "restart"})
    
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return wrap_error()
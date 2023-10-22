from fastapi import FastAPI
from chatbot import *

app = FastAPI()
chat = ChatBot()

@app.post("/message")
async def create_item(item: UserText):
    return {"message": item.message, "result": chat(item), "user_id": item.user_id}

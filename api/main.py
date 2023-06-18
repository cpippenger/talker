from typing import Union

from fastapi import FastAPI

app = FastAPI()

#chatbot = ChatBot()
import json
from conversation import Conversation
from robot import Robot

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

robot = Robot(
              name="Billy",
              persona="An old hand puppet.",
              model_name="PygmalionAI/pygmalion-6b"
              #model_name="TehVenom/Pygmalion-13b-Merged"
              #model_name="sasha0552/pygmalion-7b-f16"
              #model_file="Friendly",
              #finetune_path="/tmp/deepspeed_zero_stage2_accelerate_test/"
             )


conversation = Conversation(robot=robot)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/message/{user_id}")
def read_message(user_id: str, query: Union[str, None] = None):
    print({"user_id": user_id, "q": query})
    input_dict = json.loads(query)
    if "comment" not in input_dict:
        return {"response":"Error: Missing require input: 'comment'"}
    

    response, wav, rate = conversation.process_comment(commentor="Alec", comment=input_dict["comment"])
    return {"response":response}

          
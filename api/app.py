import json
import os,sys
import logging
import requests
from uuid import uuid4 
from flask import Flask, jsonify, request, render_template
import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import select
from sqlalchemy.sql import text
from datetime import datetime
# User imports
from models.super_chat import SuperChat

# Logging config
logging.basicConfig(
    #filename='DockProc.log',
    level=logging.INFO, 
    format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("espeakng").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DATABASE_TYPE = os.environ.get('DATABASE_TYPE',"postgresql")
DATABASE_USERNAME = os.environ.get('DATABASE_USER',"test")
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD',"test")
DATABASE_SCHEMA = os.environ.get('DATABASE_SCHEMA',"chat")
DATABASE_HOST = os.environ.get('DATABASE_HOST',"127.0.0.1")
TTS_URL=os.environ.get('TTS_URL',"http://localhost:6666/api/tts?text=")

app = Flask(__name__,static_folder="cache")
app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

SessionClass = sessionmaker(bind=engine)
session = SessionClass()

# this will get replaced with real code later
def save_tts(message,filename):
    r = requests.get(TTS_URL + message, allow_redirects=True)
    open(filename, 'wb').write(r.content)
    return None

def install():
    try:
        SuperChat.__table__.create(engine)
    except sqlalchemy.exc.ProgrammingError:
        raise

@app.get("/")
async def root():
    return render_template('index.html')

@app.route('/insert_super_chat', methods=['POST'])
def insert_super_chats():
    uuid=str(uuid4())
    data = request.json
    username=data.get('username')
    text=data.get('text')
    amount=data.get('amount')
    if username != None and text != None and amount != None:
        save_tts(username + " says " + text,"cache/" + uuid )
        session.add(SuperChat(id=uuid,username=username,text=text,amount=amount,datetime_uploaded=datetime.now()))   
        session.commit()
        return '{"status":"success"}'
    else:
        return '{"status":"failed"}'
        
@app.route('/get_super_chats')
def get_super_chats():
    statement = select(SuperChat)
    rows = session.execute(statement).all()
    output = []
    if len(rows) == 0:
        logger.warning(f"get_super_chats(): No results found.")
    # For each row
    for row in rows:
        super_chat = row._mapping["SuperChat"]
        output.append(super_chat.to_dict())
    return json.dumps(output)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "--install":
            print("Installing Database...")
            install()
        else:
            print("Unrecognized Argument")
            
    else:
        app.run(debug=True, host="0.0.0.0", port=5000)
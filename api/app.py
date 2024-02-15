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
import numpy as np
from scipy.io.wavfile import write
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
SERVICE_PORT = os.environ.get('SERVICE_PORT',5000)
SERVICE_HOST = os.environ.get('SERVICE_HOST',"0.0.0.0")
SERVICE_DEBUG = os.environ.get('SERVICE_DEBUG','True')
TTS_URL=os.environ.get('TTS_URL',"http://localhost:8100/tts")

app = Flask(__name__,static_folder="cache")
app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])






SessionClass = sessionmaker(bind=engine)
#session = SessionClass()

# this will get replaced with real code later
def save_tts(message,filename):

    try:
        r = requests.get(TTS_URL + message, allow_redirects=True)
        r.raise_for_status()
    except requests.exceptions.RequestException as err:
        return ("ERROR: OOps: Something Else") # being lazy this code sorta works
    except requests.exceptions.HTTPError as errh:
        return ("ERROR: Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        return ("ERROR: Error Connecting:")
    except requests.exceptions.Timeout as errt:
        return ("ERROR: Timeout Error:")     
    open(filename, 'wb').write(r.content)
    return filename

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
    session = SessionClass()
    uuid=str(uuid4())
    data = request.json
    username=data.get('username')
    text=data.get('text')
    amount=data.get('amount')
    if username != None and text != None and amount != None:
        
        cache_file=save_tts(username + " says " + text,"cache/" + uuid )
        
        session.add(SuperChat(id=uuid,username=username,text=text,amount=amount,datetime_uploaded=datetime.now(),tts_file=cache_file))   
        session.commit()
        session.close()
        return '{"status":"success"}'
    else:
        session.close()
        return '{"status":"failed"}'
        
@app.route('/get_super_chats', methods=['GET'])
def get_super_chats():
    session = SessionClass()
    last_seen = request.args.get('last_seen') # 2024-02-12 08:33:05
    if last_seen != None:
        statement = select(SuperChat).where(SuperChat.datetime_uploaded > last_seen ).order_by(SuperChat.datetime_uploaded.asc())
    else:
        statement = select(SuperChat).order_by(SuperChat.datetime_uploaded.asc())
    rows = session.execute(statement).all()
    session.close()
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
        app.run(debug=SERVICE_DEBUG == 'True', host=SERVICE_HOST, port=SERVICE_PORT)
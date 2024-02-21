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
import tempfile
# User imports
from models.super_chat import SuperChat
import numpy as np
from scipy.io.wavfile import write
from elevenlabs import voices, generate


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
SWC_TTS_URL=os.environ.get('SWC_TTS_URL',"http://localhost:8100/tts")
COQ_TTS_URL=os.environ.get('COQ_TTS_URL',"http://localhost:6666/api/tts?text=")

app = Flask(__name__,static_folder="cache")
app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
db = SQLAlchemy(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
SessionClass = sessionmaker(bind=engine)


# im gonna clean up the exception handling once i figure out what exceptions are what, seems weird
# cuz the repeated code is nasty
def swc_tts(str_voice,str_message,str_filename):
    payload = {'text': str_message, 'time': 'time', 'priority' : '100.0','voice_clone':str_voice}
    logger.warning(payload)
    try:
        r = requests.post(SWC_TTS_URL, json=payload)
        r.raise_for_status()
    except requests.exceptions.RequestException as err:
        return ("ERROR: " + str(err)) # being lazy this code sorta works
    except requests.exceptions.HTTPError as errh:
        return ("ERROR: swc_tts Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        return ("ERROR: swc_tts Error Connecting:")
    except requests.exceptions.Timeout as errt:
        return ("ERROR: swc_tts Timeout Error:")     
    open(str_filename, 'wb').write(r.content)    
    data=r.json()["wav"]
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(str_filename, int(r.json()["rate"]), scaled)
    return str_filename

def coq_tts(message,filename):
    try:
        r = requests.get(COQ_TTS_URL + message, allow_redirects=True)
        r.raise_for_status()
    except requests.exceptions.RequestException as err:
        return ("ERROR: coq_tts OOps: Something Else") # being lazy this code sorta works
    except requests.exceptions.HTTPError as errh:
        return ("ERROR: coq_tts Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        return ("ERROR: coq_tts Error Connecting:")
    except requests.exceptions.Timeout as errt:
        return ("ERROR: coq_tts Timeout Error:")     
    open(filename, 'wb').write(r.content)
    return filename


def eleven_tts (message,filename,voice):
    audio = generate(
          # api_key="YOUR_API_KEY", (Defaults to os.getenv(ELEVEN_API_KEY))
          text=message,
          voice=voice,
          model="eleven_multilingual_v2")
    open(filename, 'wb').write(audio)
    return filename



# - all endpoints must return path to tts file, like cache/<id>
# - return string that starts with ERROR on error 
# - can return an external url if it wants, but probably shouldnt, path
#   is passed strait to audio src
# - should be storing in /cache for playback
# - prefered to prefix file name with static id of source e.g coqtts_<id>
# - must be at least one end point


def get_swc_endpoints():
    new_endpoints={}
    # not handling exceptoins here because i want a fatal error on no connect
    r = requests.get(SWC_TTS_URL.replace('tts','get_voice_list'),allow_redirects=True)
    for voice in r.json(): 
            new_endpoints["swc_"+voice] = lambda str_message,str_filename,tmptmp=voice: swc_tts(tmptmp,str_message,str_filename)
            new_endpoints.update({
    "coq_tts": lambda str_message,str_filename: coq_tts(str_message,str_filename) ,
    "fake_tts": lambda str_message,str_filename: "cache/dummy.wav" ,
    "forced_error": lambda str_message,str_filename: "ERROR: you did this on purpose",
    "11tts_Rachel" : lambda str_message,str_filename: eleven_tts(str_message,str_filename,"Rachel")
    })
    return new_endpoints
endpoints=get_swc_endpoints()

current_endpoint="coq_tts"# hardcoded for now, not sure what to do about this,
                            # was using the first of the array
def install():
    try:
        SuperChat.__table__.create(engine)
    except sqlalchemy.exc.ProgrammingError:
        raise

@app.get("/")
async def root():
    return render_template('index.html',endpoints=endpoints.keys(),current_endpoint=current_endpoint)

@app.get("/clear-all")
async def clear_all():
    session = SessionClass()
    session.query(SuperChat).delete()
    session.commit()
    session.close()
    return '<script>location.assign("/");</script>' #hack

@app.route('/upload', methods=['POST'])
def upload_file():
    global endpoints
    # check if the post request has the file part
    if 'file' not in request.files:
        return '{"status":"MISSING FILE"}'
    else:
        file = request.files['file']
        temp_filename=tempfile.NamedTemporaryFile()  ; #pretty sure file already has a file object
        file.save(temp_filename.name)
        files={'file': (file.filename, open(temp_filename.name,'rb'))}
        r=requests.post(SWC_TTS_URL.replace('tts','upload'),files=files)
        endpoints=get_swc_endpoints()
        return r.content
    


@app.get("/update-default")
async def update_default():
    # not sure if this is good
    global current_endpoint
    endpoint= request.args.get('name')
    if endpoint != None and endpoint in endpoints.keys():
        current_endpoint=endpoint     
        return '{"status":"ok"}'
    else:
        return '{"status":"failed"}'

@app.route('/insert_super_chat', methods=['POST'])
def insert_super_chats():
    logger.warning(current_endpoint)
    session = SessionClass()
    uuid=str(uuid4())
    data = request.json
    username=data.get('username')
    text=data.get('text')
    amount=data.get('amount')
    if username != None and text != None and amount != None:
        #cache_file=save_tts(username + " says " + text,"cache/" + uuid )
        cache_file=endpoints[current_endpoint](username + " says " + text,"cache/" + uuid )
        session.add(SuperChat(id=uuid,username=username,text=current_endpoint + " - " +text,amount=amount,datetime_uploaded=datetime.now(),tts_file=cache_file))   
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
"""
Microservice that accepts text inputs and stores them in a queue. Provides a method for pulling the queue. 
"""
import os
import time
from typing import Union
from types import NoneType
import uuid
import json
import redis
import logging
import zipfile
from zipfile import ZipFile
import threading
from copy import copy
from nltk.tokenize import sent_tokenize
import numpy as np
from functools import reduce
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi import Request, Response
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import glob
# Init nltk
# TODO: Pre-init nltk in container
import nltk
nltk.download('punkt')

# User imports
from voicebox import VoiceBox

# Init redis connection
BM_TO_TTS_REDIS_HOST = os.environ['BM_TO_TTS_REDIS_HOST']
BM_TO_TTS_REDIS_PORT = os.environ.get('BM_TO_TTS_REDIS_PORT', 6379)
redis_conn = redis.Redis(host=BM_TO_TTS_REDIS_HOST, port=BM_TO_TTS_REDIS_PORT, decode_responses=True)

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


UPLOAD_FOLDER = "data"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set a lower bound on read speed, when the words per second are below this level there is a problem.
read_speed_lower_threshold = 1.5
# Set an upper bound. Often when the model skips words the read_speed will be too high.
read_speed_upper_threshold = 5.0

port = os.environ.get("READER_PORT", 8100)

# Init voicebox
voice = VoiceBox(
            logger=logger, 
            config_filename="voicebox_config.json"
)

logger.debug(f"message_queue(): Starting Message Queue service")

# Init api
app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    command: str 
    text: str
    time: str
    priority: str
    command: str = None
    is_add_start_click: bool = False
    is_add_end_click: bool = False
    speed: float = None
    voice_clone: Union[str,list] = None

# Audio samples used by TTS
audio_samples = [
    "data/gits_3.wav",
    "data/gits_2.wav",
    "data/gits_2.wav",
    "data/gits.wav"
]


data_folder = "data"

#import os
#for file in os.listdir("data/"):
#    if file.endswith(".wav"):

# TODO: Creat a more robust voice management system

voice_catalogue = {
    "major" : [
        ["data/major/major_11.wav","data/major/major_12.wav","data/major/major_13.wav","data/major/major_2_02.wav"],
        ["data/major/major_05.wav","data/major/major_06.wav","data/major/major_07.wav"],
        ["data/major/major_2_01.wav","data/major/major_2_02.wav","data/major/major_2_03.wav"],
        ["data/major/major_2_04.wav","data/major/major_2_05.wav","data/major/major_2_06.wav"],
        ["data/major/major_06","data/major/major_2_05.wav","data/major/major_12.wav"]
    ],
    "trump" : [
        ["data/trump/trump_11.wav","data/trump/trump_12.wav","data/trump/trump_13.wav","data/trump/trump_14.wav","data/trump/trump_15.wav","data/trump/trump_16.wav","data/trump/trump_17.wav","data/trump/trump_18.wav","data/trump/trump_19.wav","data/trump/trump_21.wav","data/trump/trump_22.wav","data/trump/trump_23.wav","data/trump/trump_24.wav","data/trump/trump_25.wav","data/trump/trump_26.wav","data/trump/trump_27.wav","data/trump/trump_28.wav","data/trump/trump_29.wav","data/trump/trump_30.wav",],
        ["data/trump/trump_04.wav", "data/trump/trump_05.wav", "data/trump/trump_06.wav"],
        ["data/trump/trump_07.wav", "data/trump/trump_09.wav", "data/trump/trump_10.wav"],
        ["data/trump/trump_10.wav", "data/trump/trump_15.wav", "data/trump/trump_25.wav"],
        ["data/trump/trump_07.wav", "data/trump/trump_09.wav", "data/trump/trump_13.wav"],
    ],
    "dsp" : [
        ["data/dsp/dsp_07.wav","data/dsp/dsp_12.wav","data/dsp/dsp_13.wav","data/dsp/dsp_06.wav"],
        ["data/dsp/dsp_01.wav","data/dsp/dsp_02.wav","data/dsp/dsp_03.wav","data/dsp/dsp_04.wav","data/dsp/dsp_05.wav","data/dsp/dsp_06.wav"],
        ["data/dsp/dsp_07.wav","data/dsp/dsp_12.wav","data/dsp/dsp_13.wav"],
        ["data/dsp/dsp_03.wav","data/dsp/dsp_04.wav","data/dsp/dsp_06.wav"],
        ["data/dsp/dsp_05.wav","data/dsp/dsp_07.wav","data/dsp/dsp_.wav","data/dsp/dsp_06.wav"]
    ]
 }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    #try:
    upload_filename = file.filename
    upload_name = upload_filename.replace(".zip", "")

    logger.debug(f"Reader.upload(): {upload_filename= }")
    # Save the uploaded file to the specified folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    logger.debug(f"Reader.upload(): Saving uploaded file to {file_path = }")
    try: 
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        f.close()
    except Exception as e:
        logger.debug(f"Reader.error(): Creating local file {upload_filename = }")

    extract_folder = os.path.join(UPLOAD_FOLDER, upload_name, "extracted")

    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    logger.debug(f"Reader.upload(): Extracting file at  {extract_folder = }")
    try:
        # Extract the contents of the ZIP file
        with ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder) 

    except Exception as e:
        logger.debug( "what: " + e)
    # move all wav files to root


    for path in glob.glob(extract_folder +'/**/*.wav', recursive=True):
        os.rename(path,extract_folder +'/'+ os.path.basename(path))

    # Optional: Remove the uploaded ZIP file
    os.remove(file_path)

    # Identify all wav files in the upload

    # Loop over extracted files and find WAV files
    wav_files = []
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(f"{extract_folder}/{file}")
    
    voice_catalogue[upload_name] = [wav_files]

    logger.debug(f"Reader.upload(): {voice_catalogue[upload_name]}")

    return JSONResponse(content={"message": "File uploaded and extracted successfully", "extracted_folder": extract_folder})
    #except Exception as e:
    #    # Handle any exceptions and return an appropriate response
    #    return HTTPException(status_code=500, detail=str(e))
    

@app.get("/test/")
async def test():
    logger.debug(f"TTS.test()")
    return {"message": "All good"}


@app.get("/chirp/")
async def test():
    """
    Test pushing an audio file to the redis queue.
    """
    logger.debug(f"TTS.chirp()")
    output = np.hstack((voice.click_on, voice.click_off))
    push_to_queue(output)
    return {"message": "Chirpped", "wav": json.dumps(output.tolist()), "rate" : str(24000)}

@app.get("/get_voice_list/")
async def test():
    """
    Test return list of available voices.
    """
    logger.debug(f"TTS.get_voice_list()")
    return list(voice_catalogue.keys())


@app.post("/set-config/")
async def set_config(config: dict):
    """
    Given a config dict similar to voicebox_config.json, update the live voicebox instance
    with the new config values.
    """
    logger.debug(f"MessageQueue.set-config({config = })")
    voice.config = config
    voice.synth_params = config["synth_params"]
    voice.silence_filter_params = config["silence_filter_params"]
    #voice.speaker_wav = config["vocoder"]["speaker_wav"]
    voice.vocoder = config["vocoder"]
    #voice.__init__(config=config)
    return {"message": "Config updated"}


def push_to_queue(
            wav:np.ndarray, 
            rate:int = 24050, 
            text:str = "", 
            is_normalize:bool = True, 
            priority :str="1", 
            request_time :str=None 
    ):
    """
    Given a wav and accompanying metadata, push the wav data to the audio redis queue.

    Parameters:
    -----------
    wav : np.ndarray 
        The wav values that should be sent to the audio queue.
    rate : int
        The sample rate (in hz) of the output wav. ex 16000, 22000, 24050, 48000
    text : str
        The text that was converted to audio.
    is_normalize : bool
        When true the output will be normalized to be between [-1.0, 1.0]

    Returns:
    --------
    redis_response : int
        The index in the queue of the newly inserted item. 
        If value is not a positive integer then audio was not added to the queue. 
    """
    logger.info(f"Reader.push_to_queue()")
    logger.debug(f"Reader.push_to_queue({wav = }, {rate = }, {text = }, {is_normalize})")
    # If did not receive a valid list
    if (not isinstance(wav, list)) and (not isinstance(wav, np.ndarray)):
        logger.error("push_to_queue(): Received an empty wav")

    if (isinstance(wav, list)): 
        wav = np.array(wav)

    logger.debug(f"Reader.push_to_queue(): {len(wav) = }")
    logger.debug(f"Reader.push_to_queue(): {np.min(wav) =  :.3f}")
    logger.debug(f"Reader.push_to_queue(): {np.max(wav) =  :.3f}")
    logger.debug(f"Reader.push_to_queue(): Sending chunk to queue")
    logger.debug(f"Reader.push_to_queue(): {np.mean(wav) =  :.2f}")
    logger.debug(f"Reader.push_to_queue(): {type(json_wav) = }")
    #logger.debug(f"Reader.push_to_queue(): {json_wav[0:100] = }")
    
    # If should normalize
    if is_normalize:
        logger.debug(f"Reader.push_to_queue(): Normalizing wav")
        wav = voice.normalize(wav)
        logger.debug(f"Reader.push_to_queue(): {np.min(wav) =  :.3f}")
        logger.debug(f"Reader.push_to_queue(): {np.max(wav) =  :.3f}")
        logger.debug(f"Reader.push_to_queue(): {np.mean(wav) =  :.2f}")
        #logger.debug(f"Reader.push_to_queue(): {json_wav[0:100] = }")
    
    # If received a numpy array
    if isinstance(json_wav, np.ndarray):
        # Convert numpy array to normal python list
        json_wav = wav.astype(float).tolist()
    
    # Prepare redis message
    json_wav = wav.astype(float).tolist() 
    #redis_msg = json.dumps({
    #    "data": json_wav, 
    #    "rate": rate, 
    #    "text": text, 
    #    "priority": priority, 
    #    "request_time": (str(time.time()) if (type(request_time) is not str) else request_time),
    #    "response_time": str(time.time()),
    #    "message_group_id": None,
    #    "is_last_message": True
    #})
    # Push output to audio redis queue
    #redis_response = redis_conn.lpush("audio", redis_msg)

    #return redis_response

    return 1


def process_text(
        text:str, 
        is_push_to_redis:bool=False,
        push_chunks:bool=True, 
        return_full:bool=False, 
        is_add_start_click:bool=False,  
        is_add_end_click:bool=False, 
        speed:float=None,
        priority:str=None, 
        request_time:str=None,
        should_retry:bool=True,
        voice_clone:str=None
    ):
    """
    Given a block of text, run the text through the tts model and send the resulting wav to the audio redis queue.

    Parameters:
    -----------
    test : str
        The input string that should be converted to spoken audio.
    push_chunks : bool
        A flag that specifies if the text should be broken into chunks or processed as one large text.
        With chunking each section generated will be sent to the redis queue while the next chunk is 
        processing.
    return_full : bool
        When true will return the full wav output to the client that made the request.
    is_add_start_click : bool
        When true will add a mic click on noise at the start of the generated audio.
    is_add_end_click : bool
        When true will add a mic click off noise at the end of the generated audio.
    priority : str
        The priority that the message is assigned in the initial api call. 
        Passed to the ui to control which audio clips get played at each moment. 
    request_time : str
        A timestamp of when the initial request that triggered the response to be generated was received.
        i.e. when the user speaks a command

    Returns:
    --------
    wav : list[float]
        A list of float values representing the generated wav form. The wav is single channel (mono), 32-bit float,
        24050hz sample rate. 
    """
    logger.info(f"Reader.process_text()")
    logger.debug(f"Reader.process_text({text = }, {push_chunks = }, {return_full = }, {is_add_start_click}, {is_add_end_click = }, {speed = })")
    # If given an empty string
    if not text or text.strip() == "":
        # Return nothing
        return []


    text = text.replace("'", "'")
    text = text.replace("We're", "We are")
    text = text.replace("we're", "we are")
    text = text.replace("it's", "it is")
    text = text.replace("It's", "It is")
    text = text.replace("I'll", "I will")
    text = text.replace("i'll", "i will")
    
    # If given a specific voice to clone
    if voice_clone:
        # Copy original voice
        init_speaker_wav = copy(voice.speaker_wav)
        # If is a single value
        if isinstance(voice_clone, str) and voice_clone in voice_catalogue:
            logger.debug(f"Reader.process_text(): Using {voice_clone} voice from catalogue : {voice_catalogue[voice_clone][0]}")
            voice.speaker_wav = voice_catalogue[voice_clone][0]
        # Else given some specific string
        else:    
            voice.speaker_wav = voice_clone

    # If text contains silence tokens or is long text
    if "." in text or len(text) > 250:
        #logger.debug(f"Reader.main.process_text(): Interpret pauses")
        # Split on '.'
        #texts = text.split(".")

        # Split the texts into individual sentences
        texts = sent_tokenize(text)


        # Recombine short sentences
        combined_sentences = []
        current_entry = texts[0]

        for i in range(1, len(texts)):
            current_length = len(current_entry.split())
            next_length = len(texts[i].split())
            
            if current_length + next_length <= 20:  # Adjust the threshold as needed
                current_entry += " " + texts[i]
            else:
                combined_sentences.append(current_entry.strip())
                current_entry = texts[i]

        # Add the last entry if there is any
        if current_entry:
            combined_sentences.append(current_entry.strip())
        texts = combined_sentences

        #logger.debug(f"Reader.main.process_text(): {texts = }")
        
        # Save an id to identify the group of message sent to the ui
        group_id = uuid.uuid4()

        # Save the full wav of each chunk
        full_wav = []
        # For each chunk of text separated by silence
        for chunk_index in range(len(texts)):
            logger.debug ("-"*100)
            # Extract the chunk
            chunk = texts[chunk_index].strip()
            logger.debug(f"Reader.main.process_text(): Processing {chunk = }")
            # If chunk is empty
            if not chunk or chunk.strip() == "":
                # Skip
                logger.debug(f"Reader.main.process_text(Skip blank chunk)")
                continue

            # Determine if there should be a pause in the output
            pause_length = 0.0
            # If not the last chunk to speak 
            if chunk_index < len(texts) - 1:
                # If next chunk is empty (the result of splitting a ..)
                if texts[chunk_index + 1] == "":
                    # TODO: Move to config
                    pause_length = 0.15
                else:
                    pause_length = 0.05
            #logging.debug(f"main.tts({chunk = })")

            # Determine if should add a click on or off noise to the audio chunk
            is_add_start_click_to_chunk = ((chunk_index==0) and is_add_start_click)
            is_add_end_click_to_chunk = ((chunk_index==len(texts)-2) and is_add_end_click)

            # Get wav from model
            wav, rate, wavs = voice.read_text(
                                        chunk, 
                                        is_add_start_click=is_add_start_click_to_chunk, 
                                        is_add_end_click=is_add_end_click_to_chunk,
                                        speed=speed
                                    )
            
            # Check read speed
            read_speed, read_length = voice.get_read_speed(chunk, wav)
            logger.debug(f"Reader.process_text(): init {read_speed =  :.2f}")

            # If read speed is outside valid range
            if should_retry and (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
                logger.warning(f"Reader.process_text(): Bad read speed detected : {read_speed =}")
                init_speaker_wav = copy(voice.speaker_wav)
                # Save initial and subsequent reads along with their read speed
                all_reads = [
                    {
                        "read_speed" : read_speed,
                        "wav" : wav
                    }
                ]
                # Retry generating the output with different inputs
                # TODO: Change params on each try to give a better shot of producing valid output
                retry_attempt = 1
                retry_attempts = 3
                while (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
                    
                    logger.warning(f"Reader.process_text(): Retrying generation {retry_attempt}/{retry_attempts}")
                    # Select a different set of speaker wavs
                    voice.speaker_wav = init_speaker_wav[retry_attempt]
                    # Get audio output from TTS
                    wav, rate, wavs = voice.read_text(
                                        chunk, 
                                        is_add_start_click=is_add_start_click_to_chunk, 
                                        is_add_end_click=is_add_end_click_to_chunk,
                                        speed=speed
                                    )
                    
                    read_speed, read_length = voice.get_read_speed(text = chunk, wav = wav)
                    # Save read
                    all_reads.append(
                        {
                            "read_speed" : read_speed,
                            "wav" : wav
                        }
                    )

                    logger.debug(f"Reader.process_text(): {read_speed =  :.2f}")
                    retry_attempt += 1
                    if retry_attempt > retry_attempts: 
                        break
                    if retry_attempt >= len(init_speaker_wav): 
                        break 

                # Find the best read speed in each generated sample
                logger.debug(f"Reader.process_text(): Selecting best wav out of {len(all_reads)}")
                best_read_speed = -1
                best_read_index = -1
                for read_index in range(len(all_reads)):
                    read_speed = all_reads[read_index]["read_speed"]
                    if read_speed > best_read_speed and read_speed < read_speed_upper_threshold:
                        best_read_speed = read_speed
                        best_read_index = read_index
                # If found a wav that met criteria
                if best_read_index != -1:
                    logger.debug(f"Reader.process_text(): Best wav on {read_index = } with {best_read_speed =}")
                    wav = all_reads[best_read_index]["wav"]
                else:
                    logger.debug(f"Reader.process_text(): None of the reads met criteria")


                # Reset back to original speaker wav
                voice.speaker_wav = init_speaker_wav
                
                # If still has a bad read speed
                if read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold:
                    logger.error(f"Reader.process_text(): Could not generate valid audio for {chunk = }")
                    #continue


            # If should add pause
            if pause_length > 0.0:
                # Append a pause to the wav
                wav = np.hstack((wav, np.zeros(int(24000 * pause_length), dtype=float)))
            
            # If should push individual chunks to redis
            if push_chunks:
                # Normalize individual wav
                wav = voice.normalize(wav)
                #wav_T = wav.astype(float).tolist()
                json_wav = wav.astype(float).tolist()
                #logging.debug(f"main.tts(): Received output from model: {type(wav) = }")
                # audio_queue.put(json.dumps({"data":json_wav, "rate": rate}))
                #logger.debug(f"Reader.main.process_text(): Sending chunk to queue")
                #logger.debug(f"Reader.main.process_text(): {len(wav) = }")
                #logger.debug(f"Reader.main.process_text(): {np.min(wav) = }")
                #logger.debug(f"Reader.main.process_text(): {np.max(wav) = }")
                #logger.debug(f"Reader.main.process_text(): {np.mean(wav) = }")
                #logger.debug(f"Reader.main.process_text(): {type(json_wav) = }")
                #logger.debug(f"Reader.main.process_text(): {json_wav[0:100] = }")
                
                # Check if is the last chunk being sent from the original text
                is_last_chunk = False
                # If is the last item in the list
                if (chunk_index == len(texts) - 1):
                    is_last_chunk = True
                # If is the second to last item in the list and the last item is an empty string
                if (chunk_index == len(texts) - 2) and (not texts[chunk_index + 1].strip()):
                    is_last_chunk = True

                response_time = time.time()
                #logger.debug(f"Reader.main.process_text(): {response_time =}")
                if is_push_to_redis:
                    redis_msg = json.dumps({
                                            "data": json_wav, 
                                            "rate": rate, 
                                            "text": chunk, 
                                            "priority": priority, 
                                            "request_time": request_time,
                                            "response_time": str(response_time),
                                            "message_group_id": str(group_id),
                                            "is_last_message": is_last_chunk
                                            })
                    #logger.debug(f"Reader.main.process_text(): {redis_msg = }")
                    try:
                        redis_response = redis_conn.lpush("audio", redis_msg)
                        logger.debug(f"Reader.main.process_text(): {redis_response = }")

                        # If did not get an appropriate 
                        if not redis_response or redis_response < 0:
                            logger.error("Reader.main.process_text(): Error sending audio to redis queue") 

                    except redis.exceptions.ConnectionError:
                        logger.error(f"Reader.main.process_text(): Error sending response to redis")

            # Save chunk wav
            full_wav.append(wav)

        # Recombine wavs chunks
        wav = np.hstack(full_wav)
        # Normalize
        wav = voice.normalize(wav)

    # Else just run the full text
    else:
        logger.debug(f"Reader.main.process_text(): Running full message")
        wav, rate, wavs = voice.read_text(text,
                                        is_add_start_click=is_add_start_click, 
                                        is_add_end_click=is_add_end_click,
                                        speed=speed)
        

        # Check read speed
        read_speed, read_length = voice.get_read_speed(text, wav)

        # If read speed is outside valid range
        if should_retry and (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
            # Save initial list of speakers
            init_speaker_wav = copy(voice.speaker_wav)
            logger.warning(f"Reader.process_text(): Slow read speed detected")
            # Save initial and subsequent reads along with their read speed
            all_reads = [
                {
                    "read_speed" : read_speed,
                    "wav" : wav
                }
            ]
            # Retry generating the output
            # TODO: Change params on each try to give a better shot of producing valid output
            retry_attempt = 1
            retry_attempts = 3
            while (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
                
                logger.warning(f"Reader.process_text(): Retrying generation {retry_attempt}/{retry_attempts}")
                # Select an individual audio sample
                voice.speaker_wav = init_speaker_wav[retry_attempt]
                
                wav, rate, wavs = voice.read_text(
                                    text, 
                                    is_add_start_click=False, 
                                    is_add_end_click=False,
                                    speed=speed
                                )
                
                read_speed, read_length = voice.get_read_speed(text = text, wav = wav)
                # Try fixing read speed with simple speed up
                wav, read_speed = voice.adjust_read_speed(wav, text, read_speed)
                logger.debug(f"Reader.process_text(): {read_speed = :.2f}")
                # Save read
                all_reads.append(
                    {
                        "read_speed" : read_speed,
                        "wav" : wav
                    }
                )
                retry_attempt += 1
                if retry_attempt > retry_attempts: 
                    break
                if retry_attempt >= len(init_speaker_wav): 
                    break 
            
            # Find the best read speed in each generated sample
            logger.debug(f"Reader.process_text(): Selecting best wav out of {len(all_reads)}")
            best_read_speed = -1
            best_read_index = -1
            for read_index in range(len(all_reads)):
                read_speed = all_reads[read_index]["read_speed"]
                if read_speed > best_read_speed and read_speed < read_speed_upper_threshold:
                    best_read_speed = read_speed
                    best_read_index = read_index
            # If found a wav that met criteria
            if best_read_index != -1:
                logger.debug(f"Reader.process_text(): Best wav on {read_index = } with {best_read_speed =}")
                wav = all_reads[best_read_index]["wav"]
            else:
                logger.debug(f"Reader.process_text(): None of the reads met criteria")
            # Reset back to original speaker wav
            voice.speaker_wav = init_speaker_wav
        
        # If still has a bad read speed
        if read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold:
            logger.error(f"Reader.process_text(): Could not generate valid audio for {text = }")
            #return None
        
        logger.debug(f"Reader.main.process_text(): Normalizing")
        wav = voice.normalize(wav)
        json_wav = wav.astype(float).tolist()

        if voice_clone:
            # Reset back to original speaker wav
            voice.speaker_wav = init_speaker_wav

        #logger.debug(f"Reader.main.process_text(): Sending chunk to queue")
        #logger.debug(f"Reader.main.process_text(): {len(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {np.min(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {np.max(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {np.mean(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {type(json_wav) = }")
        #logger.debug(f"Reader.main.process_text(): {json_wav[0:100] = }")

        # TODO: rename time to audio_queue time, add init_time
        response_time = time.time()
        #logger.debug(f"Reader.main.process_text(): {response_time = :.2f}")
        if is_push_to_redis:
            redis_msg = json.dumps({
                                    "data": json_wav, 
                                    "rate": rate, 
                                    "text": text, 
                                    "priority": priority, 
                                    "request_time": request_time,
                                    "response_time": str(response_time),
                                    "message_group_id": None,
                                    "is_last_message": True
                                    })
            #logger.debug(f"Reader.main.process_text(): {redis_msg = }")
            try:
                redis_response = redis_conn.lpush("audio", redis_msg)
                logger.debug(f"Reader.main.process_text(): {redis_response = }")
                # If did not get an appropriate 
                if not redis_response or redis_response < 0:
                    logger.error("Reader.main.process_text(): Error sending audio to redis queue") 
            except redis.exceptions.ConnectionError:
                logger.error(f"Reader.main.process_text(): Error sending response to redis")
        

    #run_log = {
    #    "id" : uuid4()
    #}

    return wav


def run_background_task(text: str):
    logger.debug("run_background_task(0)")
    thread = threading.Thread(target=process_text, args=(text))
    thread.start()


@app.post("/tts-stream/")
async def tts(message: Message):
    logger.debug(f"main.tts-stream({message = })")

    text = message.text
    run_background_task(text)
    
    return {"message": "Text transcribing"}


@app.post("/tts/")
async def tts(message: Message):
    logger.info(f"Reader.tts({message = })")
    # Extract text from message
    text = message.text
    is_add_start_click = message.is_add_start_click
    is_add_end_click = message.is_add_end_click
    speed = message.speed
    voice_clone = message.voice_clone

    # If given an empty string
    if not text or text.strip() == "":
        return {"message": "Error: Received an empty input string ", "wav": [], "rate" : str(24000)}

    # Get wav from tts model or load from a pre-recorded file 
    if (message.command == "weather"): 
        logger.info("Loading a pre-recorded weather command response")
        wav, rate = VoiceBox.load_wav_file_to_numpy("data/voices/trey.wav") 
        push_to_queue(wav=wav, rate=rate)

    else: 
        wav = process_text(text, is_add_start_click=is_add_start_click, is_add_end_click=is_add_end_click, priority=message.priority, request_time=message.time, speed=speed, voice_clone=voice_clone)

    logger.info(f"Reader.tts(): {type(wav)}")

    if isinstance(wav, NoneType) or (isinstance(wav, list) and len(wav) == 0):
        return {"message": "Error transcribing text", "wav": None, "rate" : str(24000)}

    # Return wav
    return {"message": "Text transcribed", "wav": wav.astype(float).tolist(), "rate" : str(24000)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

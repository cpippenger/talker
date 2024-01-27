"""
Microservice that accepts text inputs and stores them in a queue. Provides a method for pulling the queue. 
"""
import os
import time
from types import NoneType
import uuid
import json
import redis
import logging
import threading
from copy import copy
import numpy as np
from functools import reduce
from pydantic import BaseModel
from fastapi import Request, Response
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
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

# Audio samples used by TTS
audio_samples = [
    "data/gits_3.wav",
    "data/gits_2.wav",
    "data/gits.wav"
]


@app.get("/test/")
async def test():
    logger.debug(f"MessageQueue.test()")
    return {"message": "All good"}


@app.get("/chirp/")
async def test():
    """
    Test pushing an audio file to the redis queue.
    """
    logger.debug(f"MessageQueue.chirp()")
    output = np.hstack((voice.click_on, voice.click_off))
    push_to_queue(output)
    return {"message": "Chirpped", "wav": json.dumps(output.tolist()), "rate" : str(24000)}


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
    voice.speaker_wav = config["vocoder"]["speaker_wav"]
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


def get_read_speed(text, wav):
    words = nltk.tokenize.word_tokenize(text)
    word_count = 0
    for word in words:
        if word in [","]:
            continue
        word_count += 1
    audio_length = len(wav) / 24050
    words_per_second = word_count / audio_length
    return words_per_second, audio_length


def adjust_read_speed(wav, text:str, read_speed:float):
    """
    Given a wav float array, try to normalize the read speed to a target value.
    Measure the words per second spoken in the wav. Then based on the value
    apply a sliding scale of adjustments to get the audio closer to a normalized 
    read speed.

    Parameters:
    -----------
    wav : list[float|np.ndarray]
        A float array representing an audio waveform.
    text : str
        The text that the TTS model used to generate the waveform.
    read_speed : float
        The initial measured read_speed.
    
    Returns:
    --------
    wav : list[float|np.ndarray]
        The waveform with the applied speed up, if any were triggered.
    read_speed : float
        The new read speed with the speed up.
    """
    logger.info(f"Reader.adjust_read_speed()")
    logger.debug(f"Reader.adjust_read_speed(wav, {text = }, {read_speed = })")
    # If read speed is close to target
    # Apply a sliding scale speed up
    initial_read_speed = read_speed
    # Set a lower bound on read speed, when the words per second are below this level there is a problem.
    read_speed_lower_threshold = 1.5
    # Set an upper bound. Often when the model skips words the read_speed will be too high.
    read_speed_upper_threshold = 2.35
    # If read speed is close to target
    # Apply a sliding scale speed up
    if read_speed > 1.35 and read_speed < 1.4:
        logger.warning(f"Reader.adjust_read_speed(): Very Slow read speed detected")
        # Apply a another speed up
        wav = voice.apply_speed_up(wav, 1.15)
        read_speed, read_length = get_read_speed(text, wav)
        logger.warning(f"Reader.adjust_read_speed(): New read speed = {read_speed :.2f}")

    if read_speed > 1.4 and read_speed < 1.5:
        logger.warning(f"Reader.adjust_read_speed(): Slow read speed detected")
        # Apply a another speed up
        wav = voice.apply_speed_up(wav, 1.1)
        read_speed, read_length = get_read_speed(text, wav)
        logger.warning(f"Reader.adjust_read_speed(): New read speed = {read_speed :.2f}")

    if read_speed > 1.5 and read_speed < 1.55:
        logger.warning(f"Reader.adjust_read_speed(): Slightly slow read speed detected")
        # Apply a another speed up
        wav = voice.apply_speed_up(wav, 1.05)
        read_speed, read_length = get_read_speed(text, wav)
        logger.warning(f"Reader.adjust_read_speed(): New read speed = {read_speed :.2f}")

    if read_speed > read_speed_upper_threshold:
        logger.warning(f"Reader.adjust_read_speed(): Fast read speed detected")
        #wav = voice.apply_speed_up(wav, 1.32)

    if read_speed != initial_read_speed:
        logger.debug(f"Reader.adjust_read_speed(): Adjusted read speed from {initial_read_speed :.2f} to {read_speed :.2f}")

    return wav, read_speed
    

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
        should_retry:bool=False
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

    # If text contains silence tokens
    if "." in text:
        logger.debug(f"Reader.main.process_text(): Interpret pauses")
        # Split on '.'
        texts = text.split(".")
        logger.debug(f"Reader.main.process_text(): {texts = }")
        
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
                    pause_length = 0.4
                else:
                    pause_length = 0.25
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
            read_speed, read_length = get_read_speed(chunk, wav)
            logger.debug(f"Reader.process_text(): {read_speed =  :.2f}")

            # Set a lower bound on read speed, when the words per second are below this level there is a problem.
            read_speed_lower_threshold = 1.5
            # Set an upper bound. Often when the model skips words the read_speed will be too high.
            read_speed_upper_threshold = 2.35
            
            # Try fixing read speed with simple speed up
            wav, read_speed = adjust_read_speed(wav, chunk, read_speed)

            # If read speed is outside valid range
            if should_retry and (read_speed < 1.35 or read_speed > read_speed_upper_threshold):
                logger.warning(f"Reader.process_text(): Bad read speed detected")
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
                while retry_attempt <= retry_attempts and (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
                    logger.warning(f"Reader.process_text(): Retrying generation {retry_attempt}/{retry_attempts}")
                    voice.speaker_wav = audio_samples[retry_attempt]
                    # Get audio output from TTS
                    wav, rate, wavs = voice.read_text(
                                        chunk, 
                                        is_add_start_click=is_add_start_click_to_chunk, 
                                        is_add_end_click=is_add_end_click_to_chunk,
                                        speed=speed
                                    )
                    
                    read_speed, read_length = get_read_speed(text = chunk, wav = wav)
                    # Try fixing read speed with simple speed up
                    wav, read_speed = adjust_read_speed(wav, chunk, read_speed)
                    # Save read
                    all_reads.append(
                        {
                            "read_speed" : read_speed,
                            "wav" : wav
                        }
                    )

                    logger.debug(f"Reader.process_text(): {read_speed =  :.2f}")
                    retry_attempt += 1

                # Find the best read speed in each generated sample
                logger.debug(f"Reader.process_text(): Selecting best wav out of {len(all_reads)}")
                best_read_speed = -1
                best_wav = []
                for read_index in range(len(all_reads)):
                    read_speed = all_reads[read_index]["read_speed"]
                    if read_speed > best_read_speed and read_speed < read_speed_upper_threshold:
                        best_read_speed = read_speed
                        best_wav = all_reads[read_index]["wav"]
                wav = best_wav
                # Reset back to original speaker wav
                voice.speaker_wav = init_speaker_wav
                
                # If still has a bad read speed
                if read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold:
                    logger.error(f"Reader.process_text(): Could not generate valid audio for {text = }")
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
                logger.debug(f"Reader.main.process_text(): {response_time =}")
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

        return wav

    # Else just run the full text
    else:
        logger.debug(f"Reader.main.process_text(): Running full message")
        wav, rate, wavs = voice.read_text(text,
                                        is_add_start_click=is_add_start_click, 
                                        is_add_end_click=is_add_end_click,
                                        speed=speed)
        

        # Check read speed
        read_speed, read_length = get_read_speed(text, wav)

        logger.debug(f"Reader.process_text(): {read_speed = }")

        # Set a lower bound on read speed, when the words per second are below this level there is a problem.
        read_speed_lower_threshold = 1.54
        # Set an upper bound. Often when the model skips words the read_speed will be too high.
        read_speed_upper_threshold = 2.35
        
        # Try fixing read speed with simple speed up
        wav, read_speed = adjust_read_speed(wav, text, read_speed)

        # If read speed is outside valid range
        if should_retry and (read_speed < 1.35 or read_speed > read_speed_upper_threshold):
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
            while retry_attempt <= retry_attempts and (read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold):
                logger.warning(f"Reader.process_text(): Retrying generation {retry_attempt}/{retry_attempts}")
                init_speaker_wav = copy(voice.speaker_wav)
                voice.speaker_wav = audio_samples[retry_attempt]
                retry_attempt += 1
                wav, rate, wavs = voice.read_text(
                                    text, 
                                    is_add_start_click=False, 
                                    is_add_end_click=False,
                                    speed=speed
                                )
                
                read_speed, read_length = get_read_speed(text = text, wav = wav)
                # Try fixing read speed with simple speed up
                wav, read_speed = adjust_read_speed(wav, text, read_speed)
                logger.debug(f"Reader.process_text(): {read_speed = :.2f}")
                # Save read
                all_reads.append(
                    {
                        "read_speed" : read_speed,
                        "wav" : wav
                    }
                )
            
            # Find the best read speed in each generated sample
            logger.debug(f"Reader.process_text(): Selecting best wav out of {len(all_reads)}")
            best_read_speed = -1
            best_read_index = -1
            best_wav = []
            for read_index in range(len(all_reads)):
                read_speed = all_reads[read_index]["read_speed"]
                if read_speed > best_read_speed and read_speed < read_speed_upper_threshold:
                    best_read_index = read_index
                    best_read_speed = read_speed
                    best_wav = all_reads[read_index]["wav"]
                
            logger.debug(f"Reader.process_text(): Selecting best read speed = {best_read_speed :.2f} from sample {best_read_index}")
            wav = best_wav
            # Reset back to original speaker wav
            voice.speaker_wav = init_speaker_wav
        
        # If still has a bad read speed
        if read_speed < read_speed_lower_threshold or read_speed > read_speed_upper_threshold:
            logger.error(f"Reader.process_text(): Could not generate valid audio for {text = }")
            #return None
        
        logger.debug(f"Reader.main.process_text(): Normalizing")
        wav = voice.normalize(wav)
        json_wav = wav.astype(float).tolist()

        #logger.debug(f"Reader.main.process_text(): Sending chunk to queue")
        #logger.debug(f"Reader.main.process_text(): {len(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {np.min(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {np.max(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {np.mean(wav) = }")
        #logger.debug(f"Reader.main.process_text(): {type(json_wav) = }")
        #logger.debug(f"Reader.main.process_text(): {json_wav[0:100] = }")

        # TODO: rename time to audio_queue time, add init_time
        response_time = time.time()
        logger.debug(f"Reader.main.process_text(): {response_time = :.2f}")
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

    # If given an empty string
    if not text or text.strip() == "":
        return {"message": "Error: Received an empty input string ", "wav": [], "rate" : str(24000)}

    # Get wav from tts model or load from a pre-recorded file 
    if (message.command == "weather"): 
        logger.info("Loading a pre-recorded weather command response")
        wav, rate = VoiceBox.load_wav_file_to_numpy("data/voices/trey.wav") 
        push_to_queue(wav=wav, rate=rate)

    else: 
        wav = process_text(text, is_add_start_click=is_add_start_click, is_add_end_click=is_add_end_click, priority=message.priority, request_time=message.time, speed=speed)

    logger.info(f"Reader.tts(): {type(wav)}")

    if isinstance(wav, NoneType) or (isinstance(wav, list) and len(wav) == 0):
        return {"message": "Error transcribing text", "wav": None, "rate" : str(24000)}

    # Return wav
    return {"message": "Text transcribed", "wav": wav.astype(float).tolist(), "rate" : str(24000)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
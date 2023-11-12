import json
import logging
from time import time
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.api import TTS
from datasets import load_dataset
from pydub import AudioSegment
import numpy as np   
import soundfile as sf

logging.basicConfig(
     #filename='DockProc.log',
     level=logging.DEBUG, 
     format= '[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("espeakng").setLevel(logging.WARNING)


class VoiceBox():
    """
    Class used to allow easy access to various text to speech models. Can initialize models from several 
    different frameworks and allow a common interface for running inference on the model.
    """
    # TODO: Build a wrapper class around each model to avoid the branching logic
    def __init__(self, 
                logger:logging.Logger=None, 
                config:dict=None,
                config_filename:str="voicebox_config.json", 
                model_name:str=None,
                speaker_wav:str=None,
                pre_init:bool=True,
                is_persist:bool=True
                ):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel("DEBUG")
        self.logger.debug(f"{__class__.__name__}.init()")

        expose(prefix = f"{__class__.__name__}.init(): ")

        if config:
            self.config = config
        elif config_filename:
            self.config = self.load_config(config_filename)
        else:
            self.logger.error(f"{__class__.__name__}.init(): Must provide either a config dict or filename")
            return None
        
        self.logger.debug(f"{__class__.__name__}.init(): Config = {json.dumps(self.config, indent=3)}")

        # Load overrides
        # Model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.config["model"]["name"]
        # Speaker definition
        if speaker_wav:
            self.speaker_wav = speaker_wav
        else:
            self.speaker_wav = self.config["vocoder"]["speaker_wav"]

        # Save model variable
        #self.tacotron2 = None
        #self.hifi_gan = None
        self.model = None
        # If should pre-initialize the model
        if pre_init:
            self.init_model()

        self.start_time = time()        
        self.is_persist = is_persist


        self.logger.debug(f"{__class__.__name__}.init(): Complete at {self.start_time}")


    def load_config(self, config_filename:str="voicebox_config.json"):
        # Load config file
        #try:
        config = json.load(open(config_filename, "r"))
        #except:
        #    logging.error(f"{__class__.__name__}.init(): Could not read config file at {config_filename}")
        #    return None

        return config


    def save_config(self, config_filename:str="voicebox_config.json"):
        with open(config_filename, "w") as config_file:
            config_file.write(json.dumps(self.config))


    def init_model(self, model_name:str=None):
        """
        Init the listener model. 
        """

        model = None

        if not model_name:
            model_name = self.model_name
        else:
            self.model_name = model_name

        self.logger.debug(f"{__class__.__name__}.init_model(): {self.model_name = }")

        # If using fine-tuned voice model
        if self.model_name == "xtts_v1.1":
            self.logger.debug(f"{__class__.__name__}.init_model(): Init xtts model")
            # Download model
            ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v1.1")
            # Set path to model files
            # TODO: Move to config file
            model_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/model.pth"
            config_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/config.json"
            vocab_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1/vocab.json" 
            # Init config 
            self.model_config = XttsConfig()
            self.model_config.load_json(config_path)
            # Init model
            self.model = Xtts.init_from_config(self.model_config)
            self.model.load_checkpoint(
                self.model_config,
                checkpoint_path=model_path,
                vocab_path=vocab_path,
                eval=True,
                use_deepspeed=False
            )
            # If is using the gpu
            if self.config["model"]["device"] == "cuda":
                self.model.cuda()

        # If using Tacotron model
        elif self.model_name == "hf-tacotron2":
            # Read config parameters
            model_config = self.config["models"][model_name]
            model_name = self.config["model"]["name"]
            vocalizer_name = self.config["vocoder"]["name"]
            device = model_config["model"]["device"]
            
            self.logger.debug(f"{__class__.__name__}.init_model(): Loading {model_name} model")

            # Init text to speach model
            self.tacotron2 = Tacotron2.from_hparams(source=model_name, savedir="tmpdir_tts", run_opts={"device":device})
            self.hifi_gan = HIFIGAN.from_hparams(source=vocalizer_name, savedir="tmpdir_vocoder", run_opts={"device":device})

        elif self.model_name == "speecht5_tts":
            speaker_id = 1300
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(self.embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)

        #elif model_name.startswith("tts_models/"):
        #    self.tts_model = TTS(model_name) #.to(device)
            
        else:
            self.logger.error(f"{__class__.__name__}.init_model(): Unknown model name {model_name}")
        self.logger.debug(f"{__class__.__name__}.init_model(): Loading model")


    def tts(self, text:str):
        """
        Given a piece of text use the appropriate model to convert the text to a spoken audio wav.

        Parameters:
        -----------
        text : str - An input string that should be converted to audio

        Returns:
        --------
        wav : np.ndarray[float] - An array of float values representing the audio signal
        samplerate : int - An integer giving the sample rate to use when playing the audio
        """
        self.logger.debug(f"{__class__.__name__}.tts({text = })")
        self.logger.debug(f"{__class__.__name__}.tts({self.model_name = })")

        if self.model_name == "xtts_v1.1":
            
            self.logger.debug(f"{__class__.__name__}.tts(): Using xtts model with")
            self.logger.debug(f"{__class__.__name__}.tts(): params = {self.config['parameters']} ")
            outputs = self.model.synthesize(
                text,
                self.model_config,
                language="en",
                speaker_wav=self.speaker_wav,
                top_k=self.config["parameters"]["top_k"],
                top_p=self.config["parameters"]["top_p"],
                gpt_cond_len=self.config["parameters"]["gpt_conf_len"],
                decoder_iterations=self.config["parameters"]["decoder_iterations"],
                #repetition_penalty=.75
            )

            wav = outputs["wav"]
    
            sf.write("cache/test.wav", wav, 24000)
            sound = AudioSegment.from_file("cache/test.wav", format = "wav")
            sound = sound.strip_silence(silence_len=250, silence_thresh=-50, padding=100)
            wav = np.array(sound.get_array_of_samples())

            return wav, 24000
        
        # IF using Tacotron model
        elif self.model_name == "hf-tacotron2":
            self.logger.debug(f"{__class__.__name__}.tts(): Running tacotron model {self.model_name = }")
            # Running the TTS
            mel_output, mel_length, alignment = self.tacotron2.encode_text(text)
            self.logger.debug(f"{__class__.__name__}.tts(): Got model output {mel_length =}")
            # Running Vocoder (spectrogram-to-waveform)
            wav = self.hifi_gan.decode_batch(mel_output)
            wav = wav.squeeze(1)
            print (f"{type(wav) = }")
            return wav, 16000
        
        # If using T5 Model
        elif self.model_name == "speecht5_tts":
            self.logger.debug(f"{__class__.__name__}.tts(): Running t5 model {self.model_name = }")
            inputs = self.processor(text=text, return_tensors="pt")
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            return speech.numpy(), 16000

        # Disabled
        elif False and self.model_name.startswith("tts_models/"):
            self.logger.debug(f"{__class__.__name__}.tts(): Running tts model {self.model_name = }")
            tts_args = {}
            
            if self.tts_model.is_multi_lingual:
                tts_args["language"] = "en"
            
            if self.speaker_wav:
                tts_args["speaker_wav"] = self.speaker_wav
            else:
                #if self.tts_model.tts.is_multi_speakers and not self.speaker_wav:
                tts_args["speaker"] = self.tts_model.speakers[0]
            
            self.logger.debug(f"{__class__.__name__}.tts(): {tts_args =}")
            self.logger.debug(f"{__class__.__name__}.tts(): Calling model")

            wav = self.tts_model.tts(
                        text=text,
                        **tts_args
                )
            self.logger.debug(f"{__class__.__name__}.tts(): Got model results {len(wav) = }")
            return wav, 16000
        else:
            self.logger.debug(f"{__class__.__name__}.read_text(): Unknown model name {self.model_name}")
        
        return None, None


    def read_text(
            self, 
            text:str
        ):
        """
        Given a block of text use the tts model to convert to audio. Chunk the text file if necessary.
        
        Parameters:
        -----------
        text : str - A potentially large block of input text that should be converted to audio
        Returns:
        --------
        wav : np.ndarray[float] - An array of float values representing the audio signal
        samplerate : int - An integer giving the sample rate to use when playing the audio
        """
        self.logger.debug(f"{__class__.__name__}.read_text({text = }")
        
        start_time = time()
        #logging.info(f"{__class__.__name__}.{__name__}(text={text}, rate_modifier={rate_modifier}, is_say={is_say})")
        wav = None
        sample_rate = None
        # If text is long        
        if len(text) > 120:
            self.logger.debug(f"{__class__.__name__}.read_text(): Splitting text")
            #text_split = text.split(".") 
            text_split = sent_tokenize(text)
            new_text_split =  []
            for sentence in text_split:
                if len(sentence) < 120:
                    new_text_split.append(sentence)
                    continue
                sentences = sentence.split("\n")
                new_text_split.extend(sentences)
                #for new_sentence in sentences:
            text_split = new_text_split
            # Save each sentence wav
            wavs = []
            # For each sentence
            for sentence in text_split:
                self.logger.debug(f"{__class__.__name__}.read_text(): Reading {sentence =}")
                sentence = sentence.strip()
                #wav, rate = robot.read_response(sentence.strip())
                if len(sentence) < 2:
                    self.logger.debug(f"Skipping short text {len(sentence) = }")
                    #print ("Skipping")
                    continue
                try:
                    
                    self.logger.debug(f"{__class__.__name__}.read_text(): Calling self.tts")
                    wav, sample_rate = self.tts(sentence)
                    self.logger.debug(f"{__class__.__name__}.read_text(): Got results from self.tts")
                    self.logger.debug(f"{__class__.__name__}.read_text(): {type(wav) =}")
                    self.logger.debug(f"{__class__.__name__}.read_text(): {wav.shape =}")
                    
                    wavs.append(wav)
                except Exception as ex:
                    logging.error(f"{__class__.__name__}.read_text(): Failed to read text = {sentence = } ")
                    logging.error(f"{__class__.__name__}.read_text(): {ex = } ")

            self.logger.debug(f"{__class__.__name__}.read_text(): {type(wav) =}")
            self.logger.debug(f"{__class__.__name__}.read_text(): {wav.shape =}")
            self.logger.debug(f"{__class__.__name__}.read_text(): {type(wavs) = }")
            self.logger.debug(f"{__class__.__name__}.read_text(): {type(wavs[0]) = }")
            #self.logger.debug(f"{__class__.__name__}.read_text(): {wavs.shape = }")

            # Stack the wavs into a single wav
            # TODO: Add slight break between clips
            if (isinstance(wav, np.ndarray)):
                self.logger.debug(f"{__class__.__name__}.read_text(): Output is a numpy array {wav.shape }")
                wav = np.stack(np.array(wavs))
            elif (isinstance(wav, torch.Tensor)):
                wav = torch.hstack(wavs)
        
        else:
            self.logger.debug(f"{__class__.__name__}.read_text(): Running full text")
            #try:
            wav, sample_rate = self.tts(text)
            self.logger.debug(f"{__class__.__name__}.read_text(): Got model response: {len(wav) = }")
            #wav = wav.squeeze(1)
            #except:
            #    logging.error(f"{Color.F_Red}{__class__.__name__}.read_text(): Failed to read text = {text}{Color.F_White}")

        # If ran on gpu
        if "cuda" in self.config["model"]["device"]:
            # Send array back to cpu before returning
            wav = wav.cpu()

        runtime = time() - start_time
        self.logger.debug(f"{__class__.__name__}.read_text(): runtime = {runtime}")
        return wav, sample_rate



def expose(prefix:str):
    import torch
    print (f"{prefix}{torch.__version__ = }")
    import transformers
    print (f"{prefix}{transformers.__version__ = }")
    import nltk
    print (f"{prefix}{nltk.__version__ = }")
    import speechbrain
    print (f"{prefix}{speechbrain.__version__ = }")
    import TTS
    print (f"{prefix}{TTS.__version__ = }")
    import datasets
    print (f"{prefix}{datasets.__version__ = }")
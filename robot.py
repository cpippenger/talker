import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from numpy.random import random
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

from nltk.tokenize import sent_tokenize

class Robot():
    def __init__(self, 
                 name:str,
                 persona:str,
                 model_name:str="PygmalionAI/pygmalion-6b"
                ):
        print (f"Robot(name={name}, persona={persona}, model_name={model_name})")
        self.name = name
        self.persona = persona
        self.prompt_spices = ["Say it to me sexy.", "You horny puppet."]
        self.prompt_emotions = ["positive", "negative"]
        print ("Robot(): Init voice model")
        
        #models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        #    "facebook/fastspeech2-en-ljspeech",
        #    arg_overrides={"vocoder": "hifigan", "fp16": False, "cpu": True}
        #)
        #self.voice_model = models[0]
        #TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        #self.voice_generator = task.build_generator(models, cfg)
        #self.voice_task = task
        
        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
        
        print ("Robot(): Init Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print ("Robot(): Init Model")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        print ("Robot(): Setting precision to fp16")
        self.model.half()
        print ("Robot(): Send model to gpu")
        self.model.to("cuda")
        print ("Robot(): Done")
    
    
    def read_response(self, text:str, rate_modifier=None, is_say=False):
        wav = None
        # If text is long        
        if len(text) > 120:
            #text_split = text.split(".") 
            text_split = sent_tokenize(text)
            wavs = []

            for sentence in text_split:
                #print (f"Reading {sentence}")
                sentence = sentence.strip()
                #wav, rate = robot.read_response(sentence.strip())
                
                if len(sentence) < 2:
                    #print ("Skipping")
                    continue
                try:
                    # Running the TTS
                    mel_output, mel_length, alignment = self.tacotron2.encode_text(sentence)

                    # Running Vocoder (spectrogram-to-waveform)
                    wav = self.hifi_gan.decode_batch(mel_output)
                    wav = wav.squeeze(1)

                    wavs.append(wav)
                except:
                    print (f"Failed to read text = {text}")

            wav = torch.hstack(wavs)
        
        else:
            try:
                # Running the TTS
                mel_output, mel_length, alignment = self.tacotron2.encode_text(text)

                # Running Vocoder (spectrogram-to-waveform)
                wav = self.hifi_gan.decode_batch(mel_output)
                wav = wav.squeeze(1)
            except:
                print (f"Failed to read text = {text}")
            
        rate = 22050 * 1.07
        
        if rate_modifier:
            rate = rate * rate_modifier

        return wav, rate
        
    def get_robot_response(self,
                           person:str,
                           prompt:str,
                           min_len:int=128,
                           max_len:int=256
                          ):
        
        bot_input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt')
        tokenized_items = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        filter_words = ["You:", f"{self.name}:", 
                        f"{person}:", "<BOT>", 
                        "<START>", "</BOT>",
                        "Persona:",
                       "\n\n", "\n "]
        
        stopping_list = []
        for filter_word in filter_words:
            stopping_list.append(_SentinelTokenStoppingCriteria(
                sentinel_token_ids=self.tokenizer(
                    filter_word,
                    add_special_tokens=False,
                    return_tensors="pt",
                #).input_ids,
                ).input_ids.to("cuda"),
                starting_idx=tokenized_items.input_ids.shape[-1]))
            
            
        stopping_criteria_list = StoppingCriteriaList(stopping_list)
        
        logits = self.model.generate(stopping_criteria=stopping_criteria_list, 
                                     min_length=min_len+len(prompt), 
                                     max_length=max_len+len(prompt), 
                                     do_sample=True,
                                     **tokenized_items
                                    )
        # Decode output logits to words
        output = self.tokenizer.decode(logits[0], skip_special_tokens=True)
        # Filter input
        output = output[len(prompt)+1:]
        
        for filter_word in filter_words:
            if filter_word in output:
                output = output[0:output.index(filter_word)]
            
        output = output.rstrip()
        
        if "<USER>" in output:
            output = output.replace("<USER>", person)
            
        if random() > .75:
            output = f"Well {person} " + output
        
        # Remove visual expressions
        run_count = 0
        max_run_count = 5
        # Use regex to match strings like *[TEXT]* 
        match = re.search("(?<=\*)(.*?)(?=\*)", output)
        while match and run_count < max_run_count:
            run_count += 1
            output = f"{output[0:match.start()-1]}{output[match.end()+1:]}"
            match = re.search("(?<=\*)(.*?)(?=\*)", output)
        # Use regex to match strings like [[TEXT]] 
        match = re.search("(?<=\[)(.*?)(?=\])", output)
        while match and run_count < max_run_count:
            run_count += 1
            output = f"{output[0:match.start()-1]}{output[match.end()+1:]}"
            match = re.search("(?<=\*)(.*?)(?=\*)", output)
            
        # Strip whitepsace
        output = output.strip()
        
        # Remove all added prompts
        for spice in self.prompt_spices:
            output = output.replace(spice, "")
        for emotion in self.prompt_emotions:
            output = output.replace(f"Be {emotion}.\n", "")
            
        if len(output) > max_len * 2:
            print (f"Ouput too large len(output) = {len(output)}")
            output = output[0:max_len]
                
        return output
    
class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False
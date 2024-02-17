import json
import requests

# TODO: Set relative location of reader config
config = json.load(open("reader/voicebox_config.json", "r"))
config["synth_params"]["repetition_penalty"] = 1.8
config["synth_params"]["length_penalty"] = -2.5
config["synth_params"]["temperature"] = 0.00000001
config["synth_params"]["gpt_cond_len"] = 2
config["synth_params"]["top_p"] = 0.8
config["synth_params"]["top_k"] = 5
config["vocoder"]["speed_up"] = 1.27
config["vocoder"]["speaker_wav"] = "data/beth_21.wav"

r = requests.post("http://tts:8100/set-config", data=json.dumps(config))
print (r)
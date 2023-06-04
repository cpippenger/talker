import os
import time
import pickle
import IPython
import logging
import numpy as np
from copy import copy
from datetime import datetime
from numpy.random import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


from info import Info
from color import Color
from human import Human, Memory
from sentiment import Sentiment, SentimentScore
from comment import Comment


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def find_proper_nouns(sent):
    proper_nouns = []
    ner_tokens = preprocess(sent)
    for token in ner_tokens:
        if token[1] == "NNP":
            proper_nouns.append(token[0])
    # Convert list to string
    proper_nouns = " ".join(proper_nouns)
    return proper_nouns

class Conversation():
    def __init__(self, robot, is_debug=True):
        logging.info("{__class__.__name__}.__init__()")
        self.robot = robot        
        self.humans = []
        self.info = Info()
        self.chat_histories = {}
        self.chat_buffer_size = 4
        self.sentiment = Sentiment()
        self.is_debug = is_debug
        
    def __repr__(self):
        out_str =  f"Converstaion():\n"
        out_str +=  f"Robot: {self.robot.name}"
        out_str +=  f"chat_buffer_size: {self.chat_buffer_size}"
        out_str +=  f"Humans: {len(self.humans)}"
        for index in range(len(self.humans)):
            out_str +=  f"\t {index} : {self.humans[index]}"
        return out_str
    
    def save(self, filename="Conversation.p"):
        logging.info("{__class__.__name__}.save(): Save")

    def set_robot(self, robot):
        self.robot = robot
    
    def human_exists(self, name:str):
        for existing_human in self.humans:
            if existing_human.name == name:
                return True
        
    
    def get_human(self, name:str):
        for human in self.humans:
            if human.name == name:
                return human
        
    def add_human(self, human):
        if self.human_exists(human.name):
            logging.error(f"{Color.F_Red}Error: Human {human} already exists {Color.F_White}")
            return False
        self.humans.append(human)
        self.chat_histories[human.name] = ChatHistory(self.robot.name, human.name)
        
        return True
    
    def build_prompt(self,
                     commentor:str,
                     history_size:int=5
                    ):
        """
        Built an input prompt for the model given then 
        commentors history and the count of messages to include

        Parameters
        ----------
        commentor: str - The name of a user interacting with the system
        history_size: int - The number of recent messages to include in the prompt

        Returns
        -------
        prompt:str - A string containing the full prompt to send to the model encoder
        """
        logging.info(f"{__class__.__name__}.{__name__}(commentor={commentor}, history_size={history_size})")
        chat_history = copy(self.chat_histories[commentor])
        # If history too long
        if len(chat_history.dialogue) > history_size:
            # Select tail
            chat_history.dialogue = chat_history.dialogue[-history_size:]
        # Mix in human memories
        human = self.get_human(commentor)
        for index in range(len(human.positive_memories)):
            memory = human.positive_memories[index]
            # Check if memory is in chat history
            is_memory_included = False
            for comment in chat_history.dialogue:
                if comment.comment == memory.response.comment:
                    is_memory_included = True
            if not is_memory_included:
                #print(f"Inserting memory to prompt: memory = {memory}")
                chat_history.dialogue = [memory.prompt, memory.comment, memory.response] + chat_history.dialogue
        
        prompt = f"{self.robot.name}'s Persona: {self.robot.persona}\n"
        prompt += "<START>\n"
        #prompt += "[DIALOGUE HISTORY]"       
        # Dialogue history
        prompt += chat_history.prompt()
        
        # Randomly add things to prompt to steer conversation
        if random() > 0.75:
            prompt += np.random.choice(self.robot.prompt_spices) + "\n"        
        if random() > 0.5:
            prompt += f"Be {np.random.choice(self.robot.prompt_emotions)}.\n"
        
        # Robot name prompt
        prompt += f"{self.robot.name}:"
        
        return prompt
    
    def process_comment(self, 
                        commentor:str, 
                        comment:str,
                        response_length_modifier:int=0,
                        min_allowed_respose_len=2,
                        is_speak_response:bool=False
                       ):
        """
        Given a comment from a user, add the comment to the log and 
        get a response from the robot.
        
        Parameters
        ----------
        commentor : string - The name of the user making the comment
        comment : string - The comment from the 
        """
        func_name = "process_comment"
        logging.info(f"{__class__.__name__}.{func_name}({commentor}, {comment})")
        start_time = time.time()
        if not self.human_exists(commentor):
            self.add_human(Human(commentor))        
        # Convenience variables
        human = self.get_human(commentor)
        chat_history = self.chat_histories[commentor]
        
        # Preprocess text
        comment = comment.replace("...", ".. . ")
        
        # If there are any proper nouns in the text
        # search for a corresponding wiki page
        # add the summary to the context
        prompt_info = None
        proper_nouns = find_proper_nouns(comment)
        if proper_nouns:
            logging.info(f"{__class__.__name__}.{func_name}: Found proper nouns = {proper_nouns}")
            prompt_info = self.info.find_wiki_page(proper_nouns)
        
        # Check for certain trigger words
        # If asked for a long story
        if "long story" in comment:
            response_length_modifier = 1024
        
        # Get sentiment for the comment
        sentiment_dict = self.sentiment.get_sentiment(comment)
        sentiment = SentimentScore(sentiment_dict["sentiment"], sentiment_dict["positive_score"], sentiment_dict["neutral_score"], sentiment_dict["negative_score"])
        # Create comment object
        # Save comment to chat history
        chat_history.add_comment(Comment(commentor, comment, sentiment))
        # Generate robot response                
        prompt = self.build_prompt(commentor, self.chat_buffer_size)
        if prompt_info:
            prompt = prompt_info + "\n" + prompt
            logging.info(f"{__class__.__name__}.{func_name}(): Updating prompt with wiki data")

        logging.info(f"{__class__.__name__}().{func_name}: Sending prompt to robot:")
        logging.info("-"*100)
        logging.info(f"\n{Color.F_Cyan}{prompt}{Color.F_White}")
        logging.info("-"*100)

        # Randomize length of response
        min_len = 256 + int(256 * random()) + response_length_modifier
        max_len = 512 + int(1024 * random()) + response_length_modifier
        logging.info(f"{__class__.__name__}.{func_name}(): min_len = {min_len}, max_len = {max_len}")
        # Generate output from the robot given the prompt
        output = self.robot.get_robot_response(commentor, prompt, min_len=min_len, max_len=max_len)
        # Maybe retry generating output if it did not meet requirements
        retry_count = 4
        retry = 0
        should_retry = False
        if len(output) < min_allowed_respose_len:
            logging.info(f"{__class__.__name__}.{func_name}(): Output not long enough len(output) = {len(output)} ")
            should_retry = True
        while retry < retry_count and should_retry:
            logging.info(f"{__class__.__name__}.{func_name}(): Output = {output}")
            logging.info(f"{__class__.__name__}.{func_name}(): Regenerating {retry}")
            retry += 1
            output = self.robot.get_robot_response(commentor, prompt, min_len=min_len, max_len=max_len)
            should_retry = (len(output) < min_allowed_respose_len)
        
        # Text to speech output 
        wav, rate = self.robot.read_response(output)
        
        # If should speak response
        if is_speak_response:
            IPython.display.display(IPython.display.Audio(wav, rate=rate, autoplay=True))
        
        # Get sentiment for the comment
        sentiment_dict = self.sentiment.get_sentiment(output)
        sentiment = SentimentScore(sentiment_dict["sentiment"], sentiment_dict["positive_score"], sentiment_dict["neutral_score"], sentiment_dict["negative_score"])
        # Create comment object
        self.chat_histories[commentor].add_comment(Comment(self.robot.name, output, sentiment))
        
        if len(chat_history.dialogue) > 2:
            logging.info(f"{__class__.__name__}.{func_name}(): last comment = {self.chat_histories[commentor].dialogue[-2].comment}")
            logging.info(f"{__class__.__name__}.{func_name}():  sentiment = {self.chat_histories[commentor].dialogue[-2].sentiment}")
        # If has a complete set of prompt, generated response and user response
        if len(chat_history.dialogue) > 3: 
            # If the sentiment of the user response was positive
            if self.chat_histories[commentor].dialogue[-2].sentiment.sentiment == "positive":
                # Save positive interaction
                logging.info(f"{__class__.__name__}.{func_name}():  Got a positive response, saving memory")
                prompt = self.chat_histories[commentor].dialogue[-4]
                comment = self.chat_histories[commentor].dialogue[-3]
                response = self.chat_histories[commentor].dialogue[-2]
                human.add_positive_memory(Memory(prompt, comment, response))
            # If the sentiment of the user response was positive
            elif self.chat_histories[commentor].dialogue[-2].sentiment.sentiment == "negative":
                # Save negative interaction
                logging.info("f{__class__.__name__}.{func_name}():  Got a negative response, saving negative memory")
                prompt = self.chat_histories[commentor].dialogue[-4]
                comment = self.chat_histories[commentor].dialogue[-3]
                response = self.chat_histories[commentor].dialogue[-2]
                human.add_negative_memory(Memory(prompt, comment, response))

        runtime = time.time() - start_time
        logging.info(f"{__class__.__name__}.{func_name}(): runtime = {runtime}")
        return output, wav, rate
    
    
class ChatHistory():
    def __init__(self, 
                 personA:str, 
                 personB:str,
                 use_cache:bool=True
                ):
        self.cache_filename = f"{personA}-{personB}_chat_history.p"

        if use_cache and os.path.isfile(self.cache_filename):
            loaded = self.load()
            if not loaded:
                logging.error(f"{Color.F_Red}{__class__.__name__}.__init__(): Failed to load chat history {self.cache_filename}{Color.F_White}")
        else:
            # Init a fresh chat
            self.personA = personA
            self.personB = personB
            self.dialogue = [] # list of Comment objects
            self.save()


    def save(self):
        pickle.dump({"personA":self.personA, "personB":self.personB, "dialogue":self.dialogue}, open(self.cache_filename, "wb"))

        return True

    def load(self):
        if not os.path.isfile(self.cache_filename):
            logging.warning(f"{__class__.__name__}.load(): File does not exist")
            return None
        
        saved = pickle.load(open(self.cache_filename, "rb"))

        self.personA = saved["personA"]
        self.personB = saved["personB"]
        self.dialogue = saved["dialogue"] # list of Comment objects

        return True
      
    def add_comment(self, comment):
        self.dialogue.append(comment)
        self.save()
    
    def reset(self):
        self.dialogue = []
    
    def printf(self):
        out_str = ""
        for comment in self.dialogue:
            out_str += comment.printf()
            out_str += " \n"
        return out_str
    
    def prompt(self):
        out_str = ""
        for comment in self.dialogue:
            out_str += comment.prompt()
            out_str += " \n"
        return out_str
    
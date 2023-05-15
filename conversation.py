import IPython
import numpy as np
from copy import copy
from datetime import datetime
from numpy.random import random

from human import Human, Memory
from sentiment import Sentiment, SentimentScore
from comment import Comment


class Conversation():
    def __init__(self, robot, is_debug=True):
        self.robot = robot        
        self.humans = []
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
    
    def set_robot(robot):
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
            printf(f"Error: Human {name} already exists")
            return False
        self.humans.append(human)
        self.chat_histories[human.name] = ChatHistory(self.robot.name, human.name)
        
        return True
    
    def build_prompt(self,
                     commentor:str,
                     history_size:int=5
                    ):
        """"""
        chat_history = copy(self.chat_histories[commentor])
        # If history too long
        if len(chat_history.dialogue) > history_size:
            # Select tail
            chat_history.dialogue = chat_history.dialogue[-history_size:]
        # Mix in human memories
        human = self.get_human(commentor)
        for index in range(len(human.memories)):
            memory = human.memories[index]
            # Check if memory is in chat history
            is_memory_included = False
            for comment in chat_history.dialogue:
                if comment.comment == memory.response:
                    is_memory_included = True
            if not is_memory_included:
                #print(f"Inserting memory to prompt: memory = {memory}")
                chat_history.dialogue = [memory.prompt] + chat_history.dialogue
                chat_history.dialogue = [memory.comment] + chat_history.dialogue
                chat_history.dialogue = [memory.response] + chat_history.dialogue
        
        prompt = f"{self.robot.name}'s Persona: {self.robot.persona}\n"
        prompt += "<START>\n"
        #prompt += "[DIALOGUE HISTORY]"       
        # Dialogue history
        prompt += chat_history.printf()
        
        # Randomly add things to prompt to steer conversation
        if random() > 0.75:
            prompt += np.random.choice(self.robot.prompt_spices) + "\n"        
        if random() > 0.5:
            prompt += f"Be {np.random.choice(self.robot.prompt_emotions)}.\n"
        
                
        # Robot name prompt
        prompt += f"{self.robot.name}:"
        
        if self.is_debug:
            print (f"prompt:")
            print ("-"*100)
            print (prompt)
            print ("-"*100)
        
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
        #print (f"process_comment({commentor}, {comment})")
        if not self.human_exists(commentor):
            self.add_human(Human(commentor))        
        # Convenience variables
        human = self.get_human(commentor)
        chat_history = self.chat_histories[commentor]
        
        # Preprocess text
        comment = comment.replace("...", "... ")
        
        # Check for certain trigger words
        # If asked for a long story
        if "long story" in comment:
            response_length_modifier = 128
        
        # Get sentiment for the comment
        sentiment_dict = self.sentiment.get_sentiment(comment)
        sentiment = SentimentScore(sentiment_dict["sentiment"], sentiment_dict["positive_score"], sentiment_dict["neutral_score"], sentiment_dict["negative_score"])
        # Create comment object
        # Save comment to chat history
        chat_history.add_comment(Comment(commentor, comment, sentiment))
        # Generate robot response                
        prompt = self.build_prompt(commentor, self.chat_buffer_size)
        # Randomize length of response
        min_len = 128 + int(128 * random()) + response_length_modifier
        max_len = 256 + int(128 * random()) + response_length_modifier
        print (f"process_comment(): min_len = {min_len}, max_len = {max_len}")
        # Generate output from the robot given the prompt
        output = self.robot.get_robot_response(commentor, prompt, min_len=min_len, max_len=max_len)
        # Maybe retry generating output if it did not meet requirements
        retry_count = 4
        retry = 0
        should_retry = False
        if len(output) < min_allowed_respose_len:
            print (f"output not long enough len(output) = {len(output)} ")
            should_retry = True
        while retry < retry_count and should_retry:
            print (f"output = {output}")
            print (f"regenerating {retry}")
            retry += 1
            output = self.robot.get_robot_response(commentor, prompt, min_len=min_len, max_len=max_len)
        
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
        
        # Randomly save memory
        if random() > .5 and len(chat_history.dialogue) > 3:
            prompt = self.chat_histories[commentor].dialogue[-4]
            comment = self.chat_histories[commentor].dialogue[-3]
            response = self.chat_histories[commentor].dialogue[-2]
            human.add_memory(Memory(prompt, comment, response))
        
        
        return output, wav, rate
    
    
class ChatHistory():
    def __init__(self, 
                 personA:str, 
                 personB:str
                ):
        self.personA = personA
        self.personB = personB
        self.dialogue = [] # list of Comment objects
            
    def add_comment(self, comment):
        self.dialogue.append(comment)
    
    def reset(self):
        self.dialogue = []
    
    def printf(self):
        out_str = ""
        for comment in self.dialogue:
            out_str += comment.printf()
            out_str += " \n"
        return out_str
    
    def __repr__(self):
        out_str = ""
        for comment in self.dialogue:
            out_str += comment.printf()
            out_str += " \n"
        return out_str
    
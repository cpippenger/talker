from comment import Comment

class Memory():
    def __init__(self, prompt:Comment, comment:Comment, response:Comment):
        '''
            Parameters
            ----------
            prompt : Comment 
                The last section of the prompt provided to the robot
            comment : Comment
                The robot generated response to the prompt
            repsonse : Comment
                The user response to the robot comment
        '''
        self.prompt = prompt
        self.comment = comment
        self.response = response
    
    
    def __repr__(self):
        out_str = f"prompt: {self.prompt.printf()}\n"
        out_str += f"comment: {self.comment.printf()}\n"
        out_str += f"response: {self.response.printf()}\n"
        return out_str
        
    
class Human():
    def __init__(self,
                 name:str
                ):
        self.name = name
        self.positive_memories = []
        self.negative_memories = []
    
    def add_positive_memory(self, memory):
        self.positive_memories = [memory] + self.positive_memories
        
    def add_negative_memory(self, memory):
        self.negative_memories = [memory] + self.negative_memories
            
    def __repr__(self):
        return f"Human({self.name})"
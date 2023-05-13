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
        self.memories = []
    
    def add_memory(self, memory):
        self.memories = [memory] + self.memories
            
    def __repr__(self):
        return f"Human({self.name})"
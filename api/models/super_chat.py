from sqlalchemy import Column, Integer, String, DateTime, Text, UUID
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class SuperChat(Base):
    __tablename__ = 'super_chat'

    id = Column(UUID, primary_key=True, nullable=False, unique=True)
    username = Column(String(150), nullable=True, unique=False)
    text = Column(Text, nullable=True, unique=False)
    datetime_uploaded = Column(DateTime, nullable=True, unique=False)


    def __str__(self):
        return f'SuperChat(id="{str(self.id)}", username="{self.username}", text="{self.text}, datetime_uploaded="{str(self.datetime_uploaded)}'


    def to_dict(self):
        return {
            "id" : str(self.id),
            "username" : self.username,
            "text" : self.text,
            "datetime_uploaded" : str(self.datetime_uploaded)
        }
# Talker

A chatbot api that manages interacts with a HuggingFace Casual Lanugage Model. The api stores history for interactions with different users.
It is able to source external data for specfic nouns in the user input. Every comment made by the user and response generate by the bot are sent through a sentiment analysis.



# Docker

## Build - with name 'api' and tag '1.0' from api/Dockerfile_api.sh
sudo docker build --tag api:1.0 . -f ./api/Dockerfile_api

## Run the container
sudo docker run -p 80:80 api:1.0



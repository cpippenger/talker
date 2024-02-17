FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && \
    apt install -y --no-install-recommends python3-pip python3
# Set container working directory
WORKDIR /api

# Copy api files
COPY ./api/* /api/

# Install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

# Copy all user files
COPY ./info.py /api/info.py
COPY ./sentiment.py /api/sentiment.py
COPY ./conversation.py /api/conversation.py
COPY ./robot.py /api/robot.py
COPY ./color.py /api/color.py
COPY ./comment.py /api/comment.py
COPY ./human.py /api/human.py
COPY ./DatabaseFactory.py /api/DatabaseFactory.py

# Entry point to start unvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


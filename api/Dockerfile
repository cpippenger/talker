FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN chmod 777 /app/run.sh

#CMD ["python", "app.py"]
CMD ["./run.sh"]

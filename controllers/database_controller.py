import logging
import psycopg2
from models.comment import Comment

class DataBaseController():

    def __init__(self):
        self.conn = self.connect()

    def save_comment(self, comment:Comment):
        # Get cursor
        cursor = self.conn.cursor()
        # Build query
        query = "INSERT INTO comment (id, commentor, comment, time, positive_score, negative_score, sentiment) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        # Execute query
        cursor.execute(query, (comment.id, comment.commentor, comment.comment, comment.time, float(comment.sentiment["positive_score"]), float(comment.sentiment["negative_score"]), comment.sentiment["sentiment"]))
        # Commit
        self.conn.commit()

    def connect(self,conn=None,host="192.168.1.120",port="5432",username="postgres",password="xxxPASSWORDxxx",database="talker") -> bool:
        try:
            conn = psycopg2.connect(
                dbname = database,
                user = username,
                host = host,
                password = password
            )
            return conn
        except psycopg2.OperationalError as err:
            logging.info(f"{__class__.__name__}.connect(): Error connecting to db at {host=}:{port=} {username=} password {database=}")
            return None

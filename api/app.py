
import json
import logging
import pandas as pd
from uuid import uuid4
from flask import Flask, jsonify, request
import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import select
from sqlalchemy.sql import text


# User imports
from models.super_chat import SuperChat

# Logging config
logging.basicConfig(
    #filename='DockProc.log',
    level=logging.INFO, 
    format='[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("espeakng").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATABASE_TYPE = "postgresql"
DATABASE_USERNAME = "test"
DATABASE_PASSWORD = "test"
DATABASE_SCHEMA = "chat"
DATABASE_HOST = "192.168.1.4"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'{DATABASE_TYPE}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_SCHEMA}'
db = SQLAlchemy(app)
ma = Marshmallow(app)


url = URL.create(
    drivername=DATABASE_TYPE,
    username=DATABASE_USERNAME,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    database=DATABASE_SCHEMA
)

engine = create_engine(url)
Session = sessionmaker(bind=engine)

def get_db_session():
    return Session

session = get_db_session()

# Init tables
try:
    SuperChat.__table__.create(engine)
except sqlalchemy.exc.ProgrammingError:
    pass


@app.get("/test")
async def root():
    return {"message": "All good"}





@app.route('/get_super_chats')
def get_super_chats():

    statement = select(SuperChat)
    logger.debug(f"{type(statement) = }")
    logger.debug(f"{statement = }")
    #user_obj = session.scalars(statement).all()

    session = Session()
    rows = session.execute(statement).all()

    output = []

    if len(rows) == 0:
        logger.warning(f"get_super_chats(): No results found.")

    # For each row
    for row in rows:
        #logger.debug(f"{type(row) = }")
        #logger.debug(f"{row = }")
        #logger.debug(f"{row._mapping = }")
        super_chat = row._mapping["SuperChat"]
        #logger.debug(f"{type(aor) = }")
        #logger.debug(f"{aor = }")
        #logger.debug(f"{aor.to_dict() = }")
        #output.append()
        output.append(super_chat.to_dict())

    #df = pd.DataFrame.from_records(dict(zip(r.keys(), r)) for r in rows)

    
    return json.dumps(output)



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
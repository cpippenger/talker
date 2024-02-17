import os
import json
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request

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

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


documents = []


@app.get("/", response_class=HTMLResponse)
async def hello(request: Request):
    logging.debug(f"read_root()")
    context = {
        "title": "Upload a document",
        "documents": documents,
        "request": request
    }
    return templates.TemplateResponse("index.html", context=context)


@app.get("/test")
async def root():
    return {"message": "All good"}


@app.post("/upload/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    try:
        with open(f"/uploads/{file.filename}", "wb") as upload_file:
            upload_file.write(file.file.read())
        documents.append(file.filename)
    except Exception:
        print (f"Error: Writing upload file")
    finally:
        file.file.close()

    # Perform document interaction logic here
    context = {
        "title": "Upload a document",
        "documents": documents,
        "request": request
    }
    return templates.TemplateResponse("index.html", context=context)
    
import logging
import uvicorn
from typing import Optional
from fastapi import FastAPI, Form, Request, Body, UploadFile, File
# from services import service
import datetime
from typing import List
from pydantic import BaseModel, Field
import handler
app = FastAPI()


@app.post("/sbd")
async def sbd(sentence: str = Form(...),):
    """[]

    Args:
        ppt_resource_file (UploadFile, optionaduikang l): [description]. Defaults to File(...).
        ppt_json (str, optional): [description]. Defaults to Form(...).

    Returns:
        [type]: [description]
    """
    response = {"message": "", "result": ""}
    try:
        # 校验json
        response["result"] = handler.infer(text=sentence)
    except Exception as e:
        logging.error("sbd:{}".format(e.__repr__()))
        response["message"] = e.__repr__()
    return response


if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0",
                port=8101, reload=True, debug=True)

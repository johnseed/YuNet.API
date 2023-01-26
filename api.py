from typing import List, Optional
import os
from fastapi import Depends, FastAPI, HTTPException, Query, Body
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
# https://iq-inc.com/importerror-attempted-relative-import/
# import logging

from multiprocessing import Pool

import yunet_sdk
import nudenet_sdk
import insightface_sdk

# import seqlog
# seqlog.set_global_log_properties(
#     ApplicationName="YuNet",
#     Environment="Docker-Compose"
# )
# seqlog.log_to_seq(
#     server_url="http://seq",
#     level=logging.INFO,
#     batch_size=1,
#     auto_flush_timeout=1,  # seconds
#     override_root_logger=True
# )


app = FastAPI(title='YuNet', description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).',
              docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )

@app.get("/", summary='Hello World', description='Hello World', tags=['Hello World'])
def read_root():
    return {"swagger": "/docs"}

@app.post("/detect_face", summary='detect face', tags=['YuNet'], response_description="""
 the location of a face and 5 facial landmarks. The format of each row is as follows:
x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
, where x1, y1, w, h are the top-left coordinates, width and height of the face bounding box, {x, y}_{re, le, nt, rcm, lcm} stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.
"""
                                                                      "返回结果：\n\n"
                                                                      "Faces: bouding box [0:4]\n\n"
                                                                      "landmarks: 5 landmarks [4:14]\n\n"
                                                                      "confidence [-1]\n\n"
                                                                      "")
def detect_face(detectParam: yunet_sdk.DetectionParam = Body(..., description="文件路径参数", example = {
	"input": "/C/Intel/1.png",
	"output": "/C/Intel/1-faces.png",
    "save": True
}
)):
    # logging.info("sst_qa, {param}", param=qaParam.__dict__)
    if(not validate_multi_file_path([detectParam.input, detectParam.output])):
        return False
    return yunet_sdk.detect_face(detectParam)

@app.post("/recognize_face", summary='recognize face', tags=['SFace'], response_description=""
                                                                      "返回结果：\n\n"
                                                                      "confidence [0-1]\n\n"
                                                                      "")
def recognize_face(recognizeParam: yunet_sdk.RecognizeParam = Body(..., description="faces", example = {
	"feature0": [],
	"feature1": []
}
)):
    return yunet_sdk.recognize_face(recognizeParam)


# ----------------------------------------------------------------


@app.post("/detect_face_insightface", summary='detect face', tags=['InsightFace'], response_description="""
 
"""
                                                                      "返回结果：\n\n"
                                                                      "Faces: bouding box [0:4]\n\n"
                                                                      "landmarks: 5 landmarks [4:14]\n\n"
                                                                      "confidence [-1]\n\n"
                                                                      "")
def detect_face_insightface(detectParam: insightface_sdk.DetectionParam = Body(..., description="文件路径参数", example = {
	"input": "/C/Intel/1.png",
	"output": "/C/Intel/1-insightfaces.png",
    "save": True
}
)):
    # logging.info("sst_qa, {param}", param=qaParam.__dict__)
    if(not validate_multi_file_path([detectParam.input, detectParam.output])):
        return False
    return insightface_sdk.detect_face(detectParam)

@app.post("/compute_sim", summary='compute sim', tags=['InsightFace'], response_description=""
                                                                      "返回结果：\n\n"
                                                                      "confidence [0-1]\n\n"
                                                                      "")
def recognize_face(recognizeParam: insightface_sdk.RecognizeParam = Body(..., description="faces", example = {
	"feature0": [],
	"feature1": []
}
)):
    return insightface_sdk.compute_sim(recognizeParam)


# ----------------------------------------------------------------

@app.post("/detect_image", summary='detect_image', tags=['NudeNet'], response_description=""
                                                                      "返回结果：\n\n"
                                                                      "图片信息\n\n"
                                                                      "")
def detect_image(nudeParam: nudenet_sdk.NudeParam = Body(..., description="参数", example = {
	"input": "/app/data/img/1.png",
}
)):
    # logging.info("sst_qa, {param}", param=qaParam.__dict__)
    return nudenet_sdk.detect_image(nudeParam)

@app.post("/detect_video", summary='detect_video', tags=['NudeNet'], response_description=""
                                                                      "返回结果：\n\n"
                                                                      "视频信息\n\n"
                                                                      "")
def detect_video(nudeParam: nudenet_sdk.NudeParam = Body(..., description="参数", example = {
	"input": "/app/data/video/1.mp4",
}
)):
    # logging.info("sst_qa, {param}", param=qaParam.__dict__)
    return nudenet_sdk.detect_video(nudeParam)

#----------------------------------------------------


@app.post("/classify_image", summary='classify_image', tags=['NudeNet'], response_description=""
                                                                      "返回结果：\n\n"
                                                                      "图片信息\n\n"
                                                                      "")
def classify_image(nudeParam: nudenet_sdk.NudeParam = Body(..., description="参数", example = {
	"input": "/app/data/img/1.png",
}
)):
    # logging.info("sst_qa, {param}", param=qaParam.__dict__)
    return nudenet_sdk.classify_image(nudeParam)

@app.post("/classify_video", summary='classify_video', tags=['NudeNet'], response_description=""
                                                                      "返回结果：\n\n"
                                                                      "视频信息\n\n"
                                                                      "")
def classify_video(nudeParam: nudenet_sdk.NudeParam = Body(..., description="参数", example = {
	"input": "/app/data/video/2.mp4",
}
)):
    # logging.info("sst_qa, {param}", param=qaParam.__dict__)
    return nudenet_sdk.classify_video(nudeParam)



# validate file path, avoid system path or critical path
def validate_file_path(file_path: str):
    file_path = os.path.abspath(file_path)
    # check file path start with system path or critical path
    invalid_path = ["/app/server.bin", "/app/static", '/home', '/etc', "/usr", "~", "/root", "/boot", "/lib", "/lib64", "/bin", "/dev", "/proc", "/run", "/sbin", "/var", "/srv"]
    for i in invalid_path:
        if file_path.startswith(i):
            return False
    # check root path
    if(file_path == '/'):
        return False
    return True
    
# validate multi file path, avoid system path or critical path
def validate_multi_file_path(files: list):
    for file in files:
        if(not validate_file_path(file)):
            return False
    return True
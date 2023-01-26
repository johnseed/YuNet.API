import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from typing import Optional
from pydantic import BaseModel, Field

class DetectionParam(BaseModel):
    save: Optional[bool] = Field(False, description="Set “True” to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input. Default will be set to “False”.")
    recognize: Optional[bool] = Field(True, description="Set “True” to recognize face.")
    input: str = Field(..., description="Image file path")
    output: Optional[str] = Field('', description="Result file path")
    d1: Optional[str] = Field(False, description="1d result")

class RecognizeParam(BaseModel):
    feature0: list = Field(..., description="face0 128D feature")
    feature1: list = Field(..., description="face1 128D feature")

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def detect_face(param: DetectionParam):
    # If input is an image
    if param.input is not None:
        image = cv2.imread(param.input)
        faces = app.get(image)
    
        # Save results if save is true
        if param.save:
            # Draw results on the input image
            rimg = app.draw_on(image, faces)
            cv2.imwrite(param.output, rimg)
            print('Resutls saved\n')
        faceList = []
        if faces is not None:
            for face in faces:
                newFace = {
                    "bbox": face.bbox.tolist(),
                    "kps": face.kps.tolist(),
                    # "landmark_3d_68": face.landmark_3d_68.tolist(),
                    # "landmark_2d_106": face.landmark_2d_106.tolist(),
                    "pose": face.pose.tolist(),
                    # "embedding": face.embedding.tolist(),
                    "normed_embedding": face.normed_embedding.tolist(),
                    "det_score": float(face.det_score),
                    "gender": int(face.gender),
                    "age": face.age,
                    "sex": face.sex
                }
                # newFace.bbox = face.bbox.tolist()
                # newFace.kps = face.kps.tolist()
                # newFace.landmark_3d_68 = face.landmark_3d_68.tolist()
                # newFace.landmark_2d_106 = face.landmark_2d_106.tolist()
                # newFace.pose = face.pose.tolist()
                # newFace.embedding = face.embedding.tolist()
                # newFace.normed_embedding = face.normed_embedding.tolist()
                # newFace.det_score = face.det_score
                # newFace.gender = face.gender
                # newFace.age = face.age
                faceList.append(newFace)
            if param.d1:
                d1 = []
                for face in faceList:
                    d = []
                    # d += face['bbox']
                    d.append(face['bbox'][0])
                    d.append(face['bbox'][1])
                    d.append(face['bbox'][2] - face['bbox'][0])
                    d.append(face['bbox'][3] - face['bbox'][1])
                    for kp in face['kps']:
                        d += kp
                    d.append(face['det_score'])
                    d1.append(d)
                return d1
            else:
                return faceList
        else:
         return []

def compute_sim(param: RecognizeParam):
    return app.models['recognition'].compute_sim(np.asarray(param.feature0), np.asarray(param.feature1))
   
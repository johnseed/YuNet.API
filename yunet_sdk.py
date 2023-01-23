# import argparse

import numpy as np
import cv2 as cv

from yunet import YuNet

from typing import Optional
from pydantic import BaseModel, Field

backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]

class DetectionParam(BaseModel):
    save: Optional[bool] = Field(True, description="Set “True” to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input. Default will be set to “False”.")
    input: str = Field(..., description="Image file path")
    output: Optional[str] = Field('', description="Result file path")


help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Choose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
try:
    backends += [cv.dnn.DNN_BACKEND_TIMVX]
    targets += [cv.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more information.')

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--input', '-i', type=str, help='Usage: Set input to a certain image, omit if using camera.')
# parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2022mar.onnx', help="Usage: Set model type, defaults to 'face_detection_yunet_2022mar.onnx'.")
# parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
# parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
# parser.add_argument('--conf_threshold', type=float, default=0.9, help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
# parser.add_argument('--nms_threshold', type=float, default=0.3, help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
# parser.add_argument('--top_k', type=int, default=5000, help='Usage: Keep top_k bounding boxes before NMS.')
# parser.add_argument('--save', '-s', type=str, default=False, help='Usage: Set “True” to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input. Default will be set to “False”.')
# parser.add_argument('--vis', '-v', type=str2bool, default=False, help='Usage: Default will be set to “True” and will open a new window to show results. Set to “False” to stop visualizations from being shown. Invalid in case of camera input.')
# args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


# Instantiate YuNet
model = YuNet(modelPath='face_detection_yunet_2022mar.onnx',
                inputSize=[320, 320],
                confThreshold=0.9,
                nmsThreshold=0.3,
                topK=5000,
                backendId=backends[1],
                targetId=targets[1])

def detect_face(param: DetectionParam):
    # If input is an image
    if param.input is not None:
        image = cv.imread(param.input)
        h, w, _ = image.shape

        # Inference
        model.setInputSize([w, h])
        results = model.infer(image)

        # # Print results
        # print('{} faces detected.'.format(results.shape[0]))
        # for idx, det in enumerate(results):
        #     print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
        #         idx, *det[:-1])
        #     )

        # Save results if save is true
        if param.save:
            # Draw results on the input image
            image = visualize(image, results)
            cv.imwrite(param.output, image)
            print('Resutls saved\n')

        if results is not None:
            return results.tolist()
        else:
         return []

from pydantic import BaseModel, Field
from nudenet import NudeDetector, NudeClassifier


class NudeParam(BaseModel):
    input: str = Field(..., description="Image file path")


# Import module
BATCH_SIZE = 4
# initialize detector (downloads the checkpoint file automatically the first time)
detector = NudeDetector() # detector = NudeDetector('base') for the "base" version of detector.

def detect_image(param: NudeParam):
    # Detect single image
    return detector.detect(param.input)
    # fast mode is ~3x faster compared to default mode with slightly lower accuracy.
    # detector.detect('path_to_image', mode='fast')
    # Returns [{'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...]


def detect_video(param: NudeParam):
    # Detect video
    # batch_size is optional; defaults to 2
    # show_progress is optional; defaults to True
    return detector.detect_video(param.input, batch_size=BATCH_SIZE, show_progress=True, mode='fast')
    # fast mode is ~3x faster compared to default mode with slightly lower accuracy.
    # detector.detect_video('path_to_video', batch_size=BATCH_SIZE, show_progress=True, mode='fast')
    # Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
    #          "preds": {frame_i: {'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...], ....}}


# initialize classifier (downloads the checkpoint file automatically the first time)
classifier = NudeClassifier()

def detect_image(param: NudeParam):
    return classifier.classify(param.input)
    # # Classify single image
    # classifier.classify('path_to_image_1')
    # # Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}
    # # Classify multiple images (batch prediction)
    # # batch_size is optional; defaults to 4
    # classifier.classify(['path_to_image_1', 'path_to_image_2'], batch_size=BATCH_SIZE)
    # # Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY},
    # #          'path_to_image_2': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}

# Classify video
# batch_size is optional; defaults to 4
def detect_image(param: NudeParam):
    return classifier.classify_video(param.input, batch_size=BATCH_SIZE)
# Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
#          "preds": {frame_i: {'safe': PROBABILITY, 'unsafe': PROBABILITY}, ....}}
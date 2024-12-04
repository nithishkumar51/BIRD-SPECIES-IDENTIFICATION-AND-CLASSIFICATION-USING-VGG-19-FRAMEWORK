import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image
from PIL import Image as PILImage
import cv2 as cv2

MODEL_TAG = "birds_species_identification_model:latest"
height = 128
width =128

birds_species_identification_runner = bentoml.keras.get(MODEL_TAG).to_runner()

birds_species_identification_service = bentoml.Service("birds_species_identification", runners = [birds_species_identification_runner])

@birds_species_identification_service.api(input = Image(), output = NumpyNdarray())
def predict(input_img: PILImage) -> np.ndarray:
    
    input_img = cv2.resize(np.array(input_img), (height, width), interpolation = cv2.INTER_NEAREST)
    input_img = np.expand_dims(input_img, axis = 0)
    output_tensor = birds_species_identification_runner.predict.run(input_img)
    return output_tensor


import sys
import numpy as np

from skimage.io import imread
from PIL import Image
from model import get_model
from config import WEIGHT_PATH


img = imread(sys.argv[1])
img = np.expand_dims(img, 0)
model = get_model()
model.load_weights(WEIGHT_PATH)
prediction = model.predict(img)[0]/255
image = Image.fromarray(prediction.squeeze(), mode='L')
image.save('result.jpg')
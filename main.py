#Written by Logan Lane
#Python script that utilizes OpenAI's CLIP Model to find similarities between two provided images.


from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os

print("Loading Learning Model")

learningModel = SentenceTransformer('clip-ViT-B-32')

filepath = ''
images = list(glob.glob('./*.jpg'))


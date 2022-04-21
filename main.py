#Written by Logan Lane
#Python script that utilizes OpenAI's CLIP Model to find similarities between two provided images.


from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os

print("Loading Learning Model")

learningModel = SentenceTransformer('clip-ViT-B-32')

filepath = '/Users/Logan/Desktop/ComparisonML'
images = list(glob.glob('./*.jpg'))
print("Images:", len(images))

encoded = learningModel.encode([Image.open(filepath) for filepath in images], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

processed = util.paraphrase_mining_embeddings(encoded)

numSimImg = 2

threshold = 0.99
near_dup = [image for image in processed if image[0] < threshold]



for score, image_id1, image_id2 in near_dup[0:numSimImg]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(images[image_id1])
    print(images[image_id2])




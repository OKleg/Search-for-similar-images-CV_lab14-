import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
import clip
from PIL import Image
import numpy as np


def image_to_vector(image, model, device, transform):
    # Convert image 
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    # Encode using CLIP 
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy() 

def search_image(k, image, model, NN_model, df, device, transform):
    # Convert image to vector
    vector = image_to_vector(image, model, device, transform).reshape(1, -1)

    # Find the k-nearest vectors
    distances, indices = NN_model.kneighbors(vector, n_neighbors=k)
    
    # Get the paths of the k-nearest images
    paths = df.loc[indices[0], 'path']
    
    return paths.tolist()
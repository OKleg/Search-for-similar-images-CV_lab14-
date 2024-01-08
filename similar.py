try:
    import cv2
    import numpy as np
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    from sklearn.neighbors import NearestNeighbors
    import ast
    import torch
    import clip
    from PIL import Image
    print("All Modules Loaded\n torch cuda is_available ")
    print(torch.cuda.is_available())
except Exception as e:
    print("Some Modules are Missing  : {}", format(e))

 

class Similar(object):

    def __init__(self, input_image, k=5):
        self.image = input_image
        self.k = k
        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = clip.load('ViT-B/32', device=self.device) 

        # Import image db (EXISTED from IPYNB)
        self.df = pd.read_csv('./webapp/database512.csv')  
        #NearestNeighbors model
        self.NN_model  = NearestNeighbors(metric='cosine').fit(np.array(self.df['vector'].apply(ast.literal_eval).tolist()))  

        #Load NN model
        #with open('./webapp/NN_model.pkl', 'rb') as f:
        #   self.NN_model = pickle.load(f)   

    #CLIP
    def image_to_vector(self, image, model):
        # Convert image
        image = Image.fromarray(self.image)
        image = self.transform(image).unsqueeze(0).to(self.device)    

        # Encode using CLIP
        with torch.no_grad():
            image_features = model.encode_image(image)  

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 

        return image_features.cpu().numpy() 

    def search_image(self, image, model, NN_model,df, k):
        # Convert image to vector
        vector = self.image_to_vector(image, model)  

        # Find the k-nearest vectors
        distances, indices = NN_model.kneighbors(vector, n_neighbors=k) 

        # Get the paths of the k-nearest images
        paths = df.loc[indices[0], 'path']  

        return paths.tolist()   

    def run(self):
        #image = cv2.imread('./webapp/coco128/train2017/images/000000000061.jpg')

        # Search for similar images
        similar_images = self.search_image(self.image, self.model, self.NN_model, self.df, self.k)    

        return similar_images




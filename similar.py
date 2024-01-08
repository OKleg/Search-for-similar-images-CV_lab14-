try:
    import cv2
    import numpy as np
    import os
    import pandas as pd
    import pickle
    from sklearn.neighbors import NearestNeighbors
    import ast
    import torch
    import clip
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    print("All Modules Loaded\n torch cuda is_available ")
    print(torch.cuda.is_available())
except Exception as e:
    print("Some Modules are Missing  : {}", format(e))

 

class Similar(object):

    def __init__(self, input_image):
        self.image = input_image
        
        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = clip.load('ViT-B/32', device=self.device) 

        # Import image db
        self.df = pd.read_csv('./webapp/data/database.csv')  
        #NearestNeighbors model
        self.NN_model  = NearestNeighbors(metric='cosine').fit(np.array(self.df['vector'].apply(ast.literal_eval).tolist()))  

    #CLIP
    def image_to_vector(self, model):
        # Convert image
        image = Image.fromarray(self.image)
        image = self.transform(image).unsqueeze(0).to(self.device)    

        # Encode using CLIP
        with torch.no_grad():
            image_features = model.encode_image(image)  

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 

        return image_features.cpu().numpy() 

    # Step 5
    # Create pd.DataFrame for all images in path
    def create_database(self, path, model):
        data = []
        for index, filename in enumerate(os.listdir(path)):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                vector = self.image_to_vector(img, model)
                if vector is not None:
                    data.append({'index': index, 'path': os.path.join(path, filename), 'vector': vector.tolist()[0]})
        df = pd.DataFrame(data)
        return df   
    
    # Step 6

    # Save
    # with open('./webapp/models/NN_model.pkl', 'wb') as f:
        # pickle.dump(NN_model, f)    

    # Load NN model
    # with open('./webapp/models/NN_model.pkl', 'rb') as f:
        # NN_model = pickle.load(f)   

    def search_image(self, image, model, NN_model,df, k):
        # Convert image to vector
        vector = self.image_to_vector(self, image, model)  

        # Find the k-nearest vectors
        distances, indices = NN_model.kneighbors(vector, n_neighbors=k) 

        # Get the paths of the k-nearest images
        paths = df.loc[indices[0], 'path']  

        return paths.tolist()   


    def run(self):
        image = cv2.imread('./webapp/data/coco128_extend1024/images/000000000061.jpg')
        k  = 4

        #Save
        with open('./webapp/models/NN_model.pkl', 'wb') as f:
            pickle.dump(NN_model, f)    
        #Load NN model
        with open('./webapp/models/NN_model.pkl', 'rb') as f:
            NN_model = pickle.load(f)   

        # Search for similar images
        similar_images = self.search_image(self, image, self.model, self.NN_model, self.df, k)    

        return similar_images




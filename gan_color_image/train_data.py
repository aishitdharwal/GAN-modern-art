import numpy as np
import os
from PIL import Image

def load_data():
    data_path = '../data'
    files_list = os.listdir(data_path)
    train_data = []
    count=0
    for filename in files_list:
        image_path = os.path.join(data_path, filename)
        
        # anti-aliasing is a technique for minimizing the distortion artifacts known as aliasing when representing a high-resolution image at a lower resolution
        image = Image.open(image_path).resize((128, 128), Image.ANTIALIAS)
        train_data.append(np.asarray(image))
        count+=1

    train_data = np.reshape(train_data, (-1, 128, 128, 3))

    return train_data

data = load_data()
np.save('train_data.npy', data)
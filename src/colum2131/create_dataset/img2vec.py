import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from glob import glob
from torch.utils.data import DataLoader

features_dir = Path('../../../features/colum2131/')
images_dir = features_dir / 'images'

images = glob(str(images_dir / '*'))

def build_model(dim=224):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    base = efn.EfficientNetB0(input_shape=(dim,dim,3), weights='imagenet', include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    return model

def load_images(img_path):
    img = load_img(img_path, target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

model = build_model()
features = np.zeros((len(images), 1280))

n = 0
dataloader = DataLoader(images, batch_size=64, shuffle=False)
for idx, batch_path in enumerate(dataloader):
    batch_images = np.zeros((64, 224, 224, 3))
    for i,object_id in enumerate(batch_path):
        batch_images[i] = load_images(object_id)
        
    batch_preds = model.predict(batch_images)
    for i,object_id in enumerate(batch_path):
        features[n,:] = batch_preds[i]
        n += 1

df = pd.DataFrame(
    features,
    columns=[f'eff_{i}' for i in range(1280)],
    index = [img.split('\\')[-1].split('.')[0] for img in images]
)

df.to_pickle(features_dir / 'calor2vec.pickle')
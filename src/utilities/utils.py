# Libraries
from io import StringIO, BytesIO
import requests
import resource
import psutil

import random
import numpy as np
import pandas as pd

from PIL import Image

from keras import backend as K


def virtual_memory():
    memory = pd.DataFrame(list(psutil.virtual_memory()),
                          index=["total",
                                 "available",
                                 "percent",
                                 "used",
                                 "free",
                                 "active",
                                 "inactive",
                                 "wired"]).transpose()
    print("used memory: %.2f%%" % (memory["used"] / memory["total"] * 100))
    print("free memory: %.2f%%" % (memory["free"] / memory["total"] * 100))


def image_displayer(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def image_translate(image_bytes):
    img = np.uint8(np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize((224, 224))))
    return img


def uniform_train_validation_sample_batch(user_train_ratings,
                                          user_validation_ratings,
                                          item_images,
                                          validation_sample_count=1000,
                                          sample=True,
                                          batch_size=None,
                                          user_idx=None):
    """
    validation_sample_count (int): Number of not-observed items to sample to get the validation set for each user.
    """

    if batch_size is not None:
        users = range(batch_size)
    else:
        users = user_idx

    triplet_train_batch = {}
    triplet_validation_batch = {}
    for b in users:

        # training set
        if sample:
            u = random.randrange(len(user_train_ratings))
        else:
            u = b
        i = user_train_ratings[u][random.randrange(len(user_train_ratings[u]))][b'productid']
        j = random.randrange(len(item_images))
        while j in [item[b'productid'] for item in user_train_ratings[u]]:
            j = random.randrange(len(item_images))

        image_i = image_translate(item_images[i][b'imgs'])
        image_j = image_translate(item_images[j][b'imgs'])
        triplet_train_batch[u] = [image_i,
                                  image_j]

        # validation set
        #print("actual RAM used: %.3f GB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10**(-9)))
        #print("validation set for %.0f" % (u))
        i = user_validation_ratings[u][0][b'productid']
        image_i = image_translate(item_images[i][b'imgs'])

        reviewed_items = set()
        for item in user_train_ratings[u]:
            reviewed_items.add(item[b'productid'])
        reviewed_items.add(user_validation_ratings[u][0][b'productid'])

        triplet_validation_batch[u] = []
        for j in random.sample(range(len(item_images)), validation_sample_count):
            if j not in reviewed_items:
                image_j = image_translate(item_images[j][b'imgs'])
                triplet_validation_batch[u].append([image_i,
                                                    image_j])

    return triplet_train_batch, triplet_validation_batch


# Define the loss function as ln(sigmoid) according to the BPR method
# > why "-" before prediction_matrix?
#   BPR wants to maximize the loss function while Keras engine minimizes it
def softplus_loss(label_matrix, prediction_matrix):
    return K.mean(K.softplus(-prediction_matrix))


# Define the metric as AUC according to the BPR method
#
# Count the ratio of prediction value > 0
# i.e., predicting positive item score > negative item score for a user
#
# Pay attention.
# Do not use a plain integer as a parameter to switch,
# instead, pass a compatible tensor (for example create it with K.zeros_like)
def auc(label_tensor, prediction_tensor):
    return K.mean(K.switch(prediction_tensor > K.zeros_like(prediction_tensor),
                           K.ones_like(prediction_tensor),    # 1
                           K.zeros_like(prediction_tensor)))  # 0


def get_layer_index_by_name(model,
                            layer_name):
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return idx

import pandas as pd
import time
import tensorflow as tf
import os
import multiprocessing
import torch

cpus = multiprocessing.cpu_count()
num_threads = cpus
os.environ["OMP_NUM_THREADS"] = str(cpus)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpus)
os.environ["TF_NUM_INTEROP_THREADS"] = str(cpus)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


# global values
_torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print time of program
def ptp(start_time, msg: str = ''):
    print("Info:", msg, time.time() - start_time)


def get_dataset(name='food101'):
    dataset = load_dataset(name)
    return dataset


def get_ai_processor(name="openai/clip-vit-base-patch32"):
    return AutoProcessor.from_pretrained(name)


def get_ai_model(name="openai/clip-vit-base-patch32"):
    return AutoModelForZeroShotImageClassification.from_pretrained(name).to(_torch_device)


def get_classes(name='data_csv/names_of_food.csv', st=0):
    df_names = pd.read_csv(name)

    if st != 0:
        ptp(st, 'load csv of classes')

    names = list()

    for item in df_names['name']:
        names.append(item)

    return names


def ai_calculate(classes, image, processor, model):
    inputs = processor(
        text=classes,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(_torch_device)

    # get list of result after AI
    if str(_torch_device).startswith('cuda'):
        result = model(**inputs).logits_per_image.softmax(dim=1).detach().cpu().numpy().tolist()[0]
    else:
        result = model(**inputs).logits_per_image.softmax(dim=1).detach().numpy().tolist()[0]

    return result
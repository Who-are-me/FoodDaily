import pandas as pd
import time
import tensorflow as tf
import os
import multiprocessing

cpus = multiprocessing.cpu_count()
_ml = 1
num_threads = cpus * _ml
os.environ["OMP_NUM_THREADS"] = str(cpus * _ml)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpus * _ml)
os.environ["TF_NUM_INTEROP_THREADS"] = str(cpus * _ml)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from sklearn.metrics import classification_report


# print time of program
def ptp(start_time, msg: str = ''):
    print("Info:", msg, time.time() - start_time)


def work_ai(names, item, processor, model):
    inputs = processor(
        text=names,
        images=item['image'],
        return_tensors="pt",
        padding=True
    )

    return model(**inputs).logits_per_image


def slave_ai(dataset, names, start_time, processor, model, chunk = 100):
    count_trues = 0
    count_falses = 0
    counter = 1
    y_true = list()
    y_pred = list()
    image_number = 0

    file_true = open('true.csv', 'a+')

    # FIXME check
    if not os.path.exists('true.csv'):
        file_true.write('item,' + ','.join(str(x) for x in range(chunk)) + '\n')

    file_pred = open('pred.csv', 'a+')

    # FIXME check
    if not os.path.exists('pred.csv'):
        file_pred.write('item,' + ','.join(str(x) for x in range(chunk)) + '\n')

    # TODO fix this bad code
    # item -> {'image': _, 'label': _}
    for item in dataset:
        logits_per_image = work_ai(names, item, processor, model)
        lprobs = logits_per_image.softmax(dim=1).detach().numpy().tolist()[0]

        # check metrics
        y_true.append(names[item['label']])
        y_pred.append(names[lprobs.index(max(lprobs))])

        # print massage of next 100 images parsed
        if counter >= chunk:
            ptp(start_time, f'next {counter} images parsed\ntime:')
            image_number += 1

            # save metrics to file by chunk
            file_true.write(f'{image_number},' + ','.join(str(x) for x in y_true) + '\n')
            file_pred.write(f'{image_number},' + ','.join(str(x) for x in y_pred) + '\n')

            y_true = []
            y_pred = []
            counter = 1
            break
        else:
            counter += 1

        # end for
        pass

    ptp(st, "complete data")

    return y_true, y_pred


if __name__ == '__main__':
    st = time.time()

    # list -> train, validation
    dataset = load_dataset('food101')
    ptp(st, 'load dataset')

    df_names = pd.read_csv('names_of_food.csv')
    ptp(st, 'load csv of names')
    names = list()

    for item in df_names['name']:
        names.append(item)

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ptp(st, 'load processor ai')

    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    ptp(st, 'load model ai')

    len_of_dataset = len(dataset['train'])

    with open('metrics.csv', 'a+') as file:
        file.write('class,accuracy,precision,recall,f0\n')

    slave_ai(dataset['train'], names, st, processor, model, chunk=101)

    y_true = list()
    y_pred = list()

    df_true = pd.read_csv('true.csv')
    df_pred = pd.read_csv('pred.csv')

    for ind in df_true.index:
        y_true.extend(df_true.iloc[ind, df_true.columns != 'item'].values.tolist())

    for ind in df_pred.index:
        y_pred.extend(df_pred.iloc[ind, df_pred.columns != 'item'].values.tolist())

    # print(y_true)
    # print(y_pred)
    # print("#" * 50)
    # print(len(y_true))
    # print(len(y_pred))

    cls_names = [item for item in names if item in y_pred]
    # print(cls_names)

    print("#" * 50)
    print(classification_report(y_true, y_pred, target_names=cls_names))

    pass


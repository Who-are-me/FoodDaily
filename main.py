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
# from sklearn.metrics import classification_report


# print time of program
def ptp(start_time, msg: str = ''):
    print("Info:", msg, time.time() - start_time)


def work_ai(classes, image, processor, model):
    inputs = processor(
        text=classes,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # get list of result after ai
    return model(**inputs).logits_per_image.softmax(dim=1).detach().numpy().tolist()[0]


def work_ai_by_item(classes, item, processor, model):
    inputs = processor(
        text=classes,
        images=item['image'],
        return_tensors="pt",
        padding=True
    )

    # get list of result after ai
    return model(**inputs).logits_per_image.softmax(dim=1).detach().numpy().tolist()[0]


def slave_ai(dataset, classes: list, start_time, processor, model, file_prefix, chunk = 100):
    counter = 1
    true = list()
    pred = list()
    image_number = 0
    true_file_name = file_prefix + '_true.csv'
    pred_file_name = file_prefix + '_pred.csv'
    # fix class names
    fixed_classes = [cl.replace('_', ' ') for cl in classes]

    if os.path.exists(true_file_name):
        file_true = open(true_file_name, 'a+')
    else:
        file_true = open(true_file_name, 'a+')
        file_true.write('item,' + ','.join(str(x) for x in range(chunk)) + '\n')

    if os.path.exists(pred_file_name):
        file_pred = open(pred_file_name, 'a+')
    else:
        file_pred = open(pred_file_name, 'a+')
        file_pred.write('item,' + ','.join(str(x) for x in range(chunk)) + '\n')

    # item -> {'image': _, 'label': _}
    for item in dataset:
        # magic
        lprobs = work_ai_by_item(fixed_classes, item, processor, model)

        # check metrics
        true.append(classes[item['label']])
        pred.append(classes[lprobs.index(max(lprobs))])

        # print massage of next {chunk} images parsed
        if counter >= chunk:
            ptp(start_time, f'next {counter} images parsed\ntime:')
            image_number += 1

            # save metrics to file by chunk
            file_true.write(f'{image_number},' + ','.join(str(x) for x in true) + '\n')
            file_pred.write(f'{image_number},' + ','.join(str(x) for x in pred) + '\n')

            true = []
            pred = []
            counter = 1
            exit()
        else:
            counter += 1

        # end for
        pass

    ptp(st, f"complete {file_prefix} data")

    return true_file_name, pred_file_name


def get_ai_processor(name="openai/clip-vit-base-patch32"):
    return AutoProcessor.from_pretrained(name)


def get_ai_model(name="openai/clip-vit-base-patch32"):
    return AutoModelForZeroShotImageClassification.from_pretrained(name)


def get_classes(name='csv_data/names_of_food.csv', st=0):
    df_names = pd.read_csv(name)

    if st != 0:
        ptp(st, 'load csv of classes')

    names = list()

    for item in df_names['name']:
        names.append(item)

    return names


if __name__ == '__main__':
    st = time.time()

    # list -> train, validation
    dataset = load_dataset('food101')
    ptp(st, 'load dataset')

    classes = get_classes()

    processor = get_ai_processor()
    ptp(st, 'load processor ai')

    model = get_ai_model()
    ptp(st, 'load model ai')
    # exit()

    # len_of_dataset = len(dataset['train'])

    # ? metrics
    # with open('metrics.csv', 'a+') as file:
    #     file.write('class,accuracy,precision,recall,f0\n')

    # slave_ai(dataset['train'], names, st, processor, model, file_prefix='train', chunk=750)
    slave_ai(dataset['validation'], classes, st, processor, model, file_prefix='validation', chunk=250)

    # y_true = list()
    # y_pred = list()

    # df_true = pd.read_csv('true.csv')
    # df_pred = pd.read_csv('pred.csv')

    # for ind in df_true.index:
    #     y_true.extend(df_true.iloc[ind, df_true.columns != 'item'].values.tolist())

    # for ind in df_pred.index:
    #     y_pred.extend(df_pred.iloc[ind, df_pred.columns != 'item'].values.tolist())

    # cls_names = [item for item in names if item in y_pred]

    # print("#" * 50)
    # print(classification_report(y_true, y_pred, target_names=cls_names))

    pass


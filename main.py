import pandas as pd
import time
import tensorflow as tf
import os
import multiprocessing
# import sys
# import threading

# class ReturnValueThread(threading.Thread):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.result = None
#
#     def run(self):
#         if self._target is None:
#             return  # could alternatively raise an exception, depends on the use case
#         try:
#             self.result = self._target(*self._args, **self._kwargs)
#         except Exception as exc:
#             print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception
#
#     def join(self, *args, **kwargs):
#         super().join(*args, **kwargs)
#         return self.result

cpus = multiprocessing.cpu_count()

num_threads = cpus * 2
os.environ["OMP_NUM_THREADS"] = str(cpus * 2)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpus * 2)
os.environ["TF_NUM_INTEROP_THREADS"] = str(cpus * 2)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

# from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


# print time of program
def ptp(start_time, msg: str = ''):
    print("Time:", msg, time.time() - start_time)


def work_ai(names, item, processor, model):
    inputs = processor(
        text=names,
        images=item['image'],
        return_tensors="pt",
        padding=True
    )

    return model(**inputs)


def slave(dataset, names, start_time, processor, model):
    count_trues = 0
    count_falses = 0
    counter = 1
    # items = [None, None]

    # TODO fix this bad code
    # item -> {'image': _, 'label': _}
    for item in dataset:
        outputs = work_ai(names, item, processor, model)
        logits_per_image = outputs.logits_per_image
        lprobs = logits_per_image.softmax(dim=1).detach().numpy().tolist()[0]

        if lprobs.index(max(lprobs)) == item['label']:
            count_trues += 1
        else:
            count_falses += 1

        if counter == 100:
            ptp(start_time, f'next 100 images parsed\n'
                            f'true: {count_trues}\n'
                            f'false: {count_falses}\n'
                            f'accuracy: {count_trues / (count_trues + count_falses)}\ntime:')
            counter = 1
        else:
            counter += 1

        # code for 2 threads
        # if items[0] == None:
        #     items[0] = item
        # else:
        #     items[1] = item
        #     outputs_0 = ReturnValueThread(target=work_ai, args=(names, items[0], processor, model))
        #     outputs_1 = ReturnValueThread(target=work_ai, args=(names, items[1], processor, model))
        #
        #     outputs_0.start()
        #     outputs_1.start()
        #
        #     logits_per_image_0 = outputs_0.join().logits_per_image
        #     logits_per_image_1 = outputs_1.join().logits_per_image
        #
        #     probs_0 = logits_per_image_0.softmax(dim=1)
        #     probs_1 = logits_per_image_1.softmax(dim=1)
        #     lprobs_0 = probs_0.detach().numpy().tolist()[0]
        #     lprobs_1 = probs_1.detach().numpy().tolist()[0]
        #
        #     if lprobs_0.index(max(lprobs_0)) == items[0]['label']:
        #         count_trues += 1
        #     else:
        #         count_falses += 1
        #
        #     if lprobs_1.index(max(lprobs_1)) == items[1]['label']:
        #         count_trues += 1
        #     else:
        #         count_falses += 1
        #
        #     if counter == 100:
        #         ptp(start_time, f'next 100 images parsed\n'
        #                 f'true: {count_trues}\n'
        #                 f'false: {count_falses}\n'
        #                 f'accuracy: {count_trues / (count_trues + count_falses)}\ntime:')
        #         counter = 2
        #     else:
        #         counter += 2
        #
        #     items[0] = None
        #     items[1] = None

        # end for
        pass

    ptp(st, "complete data")

    return count_trues, count_falses, count_trues / (count_trues + count_falses)


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

    count_trues, count_falses, accuracy = slave(dataset['train'], names, st, processor, model)

    print("#" * 50)
    print(f"Simple accuracy of data: {count_trues / len_of_dataset}")
    print(f"count_trues:", count_trues)
    print(f"count_falses:", count_falses)

    pass


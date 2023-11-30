from PIL import Image
import pandas as pd
import time
import os
import requests
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


# print time of program
def ptp(start_time, msg: str = ''):
    print("Time:", msg, time.time() - start_time)


if __name__ == '__main__':
    st = time.time()

    # load dataset
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

    count_true_train = 0
    count_false_train = 0

    # for 'train' data
    # item -> {'image': _, 'label': _}
    for item in dataset['train']:
        inputs = processor(
            text=names,
            images=item['image'],
            return_tensors="pt",
            padding=True
        )

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        lprobs = probs.detach().numpy().tolist()

        if lprobs.index(max(probs)) == item['label']:
            count_true_train += 1
        else:
            count_false_train += 1

        # print(outputs)
        # print(logits_per_image)
        # print(probs.detach().numpy()[0])

        pass

    count_true_validation = 0
    count_false_validation = 0

    # for 'validation' data
    # item -> {'image': _, 'label': _}
    for item in dataset['validation']:
        inputs = processor(
            text=names,
            images=item['image'],
            return_tensors="pt",
            padding=True
        )

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        lprobs = probs.detach().numpy().tolist()

        if lprobs.index(max(probs)) == item['label']:
            count_true_train += 1
        else:
            count_false_train += 1
        pass

    # print(len(dataset['train']))
    #
    # print("Time: ", time.time() - tm)
    #im =
    # dataset['train'][0]['image'].save('image.jpeg', format='jpeg')
    # print("Time: ", time.time() - tm)
    # print("Image")
    # print("Time: ", time.time() - tm)
    # print(im)

    # im.save('image.jpeg', format='jpeg')

    # print(dataset.column_names)
    # print(_NAMES[dataset['train'][987].get('label')])

    # print(dataset.class_encode_column('label') )
    # print(type(dataset))



    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # inputs = processor(
    #     text=["dog", "cat", "tiger"],
    #     images=image,
    #     return_tensors="pt",
    #     padding=True
    # )

    # outputs = model(**inputs)
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    # print(outputs)
    # print(logits_per_image)
    # print(probs.detach().numpy()[0])


    pass


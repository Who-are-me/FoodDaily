from PIL import Image
import pandas as pd
import time
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
    counter = 1

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
        lprobs = probs.detach().numpy().tolist()[0]

        if lprobs.index(max(lprobs)) == item['label']:
            count_true_train += 1
        else:
            count_false_train += 1

        if counter == 100:
            ptp(st, f'next 100 images parsed\n'
                    f'true: {count_true_train}\n'
                    f'false: {count_false_train}\n'
                    f'accuracy: {count_true_train / (count_true_train + count_false_train)}\ntime:')
            counter = 1
        else:
            counter += 1

        pass

    ptp(st, "complete 'train' data")

    count_true_validation = 0
    count_false_validation = 0
    counter = 1

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
        lprobs = probs.detach().numpy().tolist()[0]

        if lprobs.index(max(lprobs)) == item['label']:
            count_true_validation += 1
        else:
            count_false_validation += 1

        if counter == 100:
            ptp(st, f'next 100 images parsed\n'
                    f'true: {count_true_validation}\n'
                    f'false: {count_false_validation}\n'
                    f'accuracy: {count_true_validation / (count_true_validation + count_false_validation)}\ntime:')
            counter = 1
        else:
            counter += 1

        pass

    ptp(st, "complete 'validation' data")

    print("#" * 50)
    print(f"Simple accuracy of train data: {count_true_train / len(dataset['train'])}")
    print(f"count_true_train:", count_true_train)
    print(f"count_false_train:", count_false_train)

    print("#" * 50)
    print(f"Simple accuracy of validation data: {count_true_validation / len(dataset['validation'])}")
    print(f"count_true_validation:", count_true_validation)
    print(f"count_false_validation:", count_false_validation)

    pass


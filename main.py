from src.interface_ai import *


def preparation_data(dataset, classes: list, processor, model, file_prefix, chunk=100):
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

    time_now = time.time()
    # item -> {'image': _, 'label': _}
    for item in dataset:
        # magic
        list_probs = ai_calculate(fixed_classes, item['image'], processor, model)

        # check metrics
        true.append(classes[item['label']])
        pred.append(classes[list_probs.index(max(list_probs))])

        # print massage of next {chunk} images parsed
        if counter >= chunk:
            ptp(time_now, f'next {counter} images parsed\ntime:')
            time_now = time.time()
            image_number += 1

            # save metrics to file by chunk
            file_true.write(f'{image_number},' + ','.join(str(x) for x in true) + '\n')
            file_pred.write(f'{image_number},' + ','.join(str(x) for x in pred) + '\n')

            true = []
            pred = []
            counter = 1
            # exit()
        else:
            counter += 1

        # end for
        pass

    ptp(st, f"complete {file_prefix} data")

    return true_file_name, pred_file_name


if __name__ == '__main__':
    st = time.time()

    # list -> train, validation
    dataset = get_dataset()
    ptp(st, 'load dataset')

    classes = get_classes()

    processor = get_ai_processor()
    ptp(st, 'load processor ai')

    model = get_ai_model()
    ptp(st, 'load model ai')

    # len_of_dataset = len(dataset['train'])

    # ? metrics
    # with open('metrics.csv', 'a+') as file:
    #     file.write('class,accuracy,precision,recall,f0\n')

    # slave_ai(dataset['train'], names, st, processor, model, file_prefix='train', chunk=750)
    preparation_data(dataset['validation'], classes, processor, model, file_prefix='validation', chunk=250)

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


csv_file = open('data.csv', 'w+')
csv_file.write('item,true,false,accuracy,precision,recall,f0,accuracy_sum\n')

with open('console_output.txt', 'r+') as file:
    item = 0
    trues = 0.0
    ltrues = 0.0
    falses = 0.0
    lfalses = 0.0

    accuracy = lambda ts, fs: ts / (ts + fs)
    precision = lambda tp, fp: tp / (tp + fp)
    recall = lambda tp, fn: tp / (tp + fn)
    fb = lambda _pre, _rec, b: (1 + b ** 2) * ((_pre * _rec) / ((b ** 2 * _pre) + _rec))

    for line in file:
        if line.startswith('true: '):
            trues =  float(line[6:]) - ltrues
            ltrues = float(line[6:])
        elif line.startswith('false: '):
            falses = float(line[7:]) - lfalses
            lfalses = float(line[7:])
        elif line.startswith('accuracy: '):
            csv_file.write(','.join([str(item),
                                     str(trues),
                                     str(falses),
                                     str(accuracy(trues,falses)),
                                     str(precision(trues,falses)),
                                     str(recall(trues,0)),
                                     str(fb(precision(trues,falses),recall(trues,falses),0)),
                                     str(line[10:])
                                     ]) + '\n')
            item += 1
        elif line.startswith("Time: complete"):
            break

        pass

import pandas as pd
from sklearn.metrics import classification_report


df_names = pd.read_csv('names_of_food.csv')
names = list()

for item in df_names['name']:
    names.append(item)

y_true = list()
y_pred = list()
# y_true_v = list()
# y_pred_v = list()

df_true = pd.read_csv('true.csv')
df_pred = pd.read_csv('pred.csv')

# df_true_v = df_true.iloc[751:]
# df_pred_v = df_pred.iloc[751:]

df_true = df_true.iloc[:750]
df_pred = df_pred.iloc[:750]

for ind in df_true.index:
    y_true.extend(df_true.iloc[ind, df_true.columns != 'item'].values.tolist())

for ind in df_pred.index:
    y_pred.extend(df_pred.iloc[ind, df_pred.columns != 'item'].values.tolist())

cls_names = [item for item in names if item in y_pred]

# for ind in df_true_v.index:
#     y_true_v.extend(df_true_v.iloc[ind, df_true_v.columns != 'item'].values.tolist())

# for ind in df_pred_v.index:
#     y_pred_v.extend(df_pred_v.iloc[ind, df_pred_v.columns != 'item'].values.tolist())

# cls_names_v = [item for item in names if item in y_pred_v]

report_train = classification_report(y_true, y_pred, target_names=cls_names, output_dict=True)
df_report_train = pd.DataFrame(report_train).transpose()
df_report_train.to_csv('report_train.csv')

print("#" * 70)
print("METRICS: train data")
print(classification_report(y_true, y_pred, target_names=cls_names))

# print("#" * 70)
# print("METRICS: validation data")
# print(classification_report(y_true_v, y_pred_v, target_names=cls_names_v))

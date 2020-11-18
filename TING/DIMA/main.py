print("Программа DIMA начала работу!\n")


# Подключаем библиотiеки:
#
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from ProcessDataset import process_dataset
from Search import D_search
from Search import R_search
from Search import B_search


# Инициализируем название выходного файла:
#
result_name = 'Feature_importance.xlsx'


# Название файла:
#
xlsx_name = input('Названия файла: ')
df = pd.read_excel(xlsx_name, date_parser="Дата", index_col="Дата")


# Характеристика фичей:
#
targetName = "Дебит жидкости"
lag_max = 30
columns_in = {'Динамический уровень': True,\
              'Приемистость воды': True,\
              'Обводненность': False}

columns_types = []
for pair in columns_in.items():
  if pair[1]: columns_types.append(pair[0])

calendar_features = True
statistic_features = False


# Тип поиска важных признаков:
#
FImpType = input('Какой тип поиска признаков: ') # ['D', 'R', 'B']

if FImpType in ['R', 'B']:
    TreeNumber = int(input('Кол-во деревьев в лесу: '))


# Препарация датасета:
#
dataset = process_dataset(dataset = df,
                          targetName = targetName,
                          lag_max = lag_max,
                          column_types = columns_types,
                          statistic_features = statistic_features,
                          calendar_features = calendar_features)

print('\nРазмеры таблицы размноженного датасета.')
print('Кол-во строк:\t', dataset.shape[0])
print('Кол-во столбцов:', dataset.shape[1], end='\n\n')


# Запоминаем какое название имеет таргет:
#
target_name = dataset.columns[0]

if   FImpType == 'D':
    feat_df = D_search(dataset, target_name)

elif FImpType == 'R':
    feat_df = R_search(dataset, 
                       target_name, 
                       TreeNumber)

elif FImpType == 'B':
    feat_df = B_search(dataset, 
                       target_name,
                       TreeNumber)

feat_df.to_excel(result_name)

print("Программа DIMA завершила работу!")

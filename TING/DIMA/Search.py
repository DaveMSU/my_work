import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt # Для графиков.
import seaborn as sns  # Для красивых графиков.


# Фиксации seed'ов и более удобного вывода:
import random
import time
from tqdm import tqdm_notebook as tqdm


# Для декомпзиции временных рядов:
import scipy.stats     as stats # Библиотека с функциями матстата и теорвера.
import statsmodels.api as sm    # Для STL - декомпозиции временных рядов.
from statsmodels.tsa.api     import ExponentialSmoothing # baseline предикт-модель.
from sklearn.model_selection import TimeSeriesSplit      # Для настройки кросс-валидации.


# Модели машинного обучения:
from sklearn.ensemble import RandomForestRegressor     #!
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm      import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors    import KNeighborsRegressor


# Boruta:
from boruta import BorutaPy


# Фуксируем seed:
seed_num = 1

random.seed(seed_num)
np.random.seed(seed_num)





def D_search(dataset,
             target_name):

  all_target_corr = dataset.corr().loc[:,[target_name]]**2
  all_target_corr = all_target_corr.sort_values(target_name, ascending=False).drop(target_name)**0.5

  feat_df = pd.DataFrame({'feature': all_target_corr.index,
                          'value':   all_target_corr.values[:,0]})

  return feat_df







def R_search(dataset,
             target_name,
             TreeNumber):

  def get_fnames(TreeNumber):
      X_tmp = dataset.iloc[:,1:].values
      y_tmp = dataset.iloc[:,0].values

      model = RandomForestRegressor(TreeNumber) # Больше деревьев, но точней результат.
      model.fit(X_tmp, y_tmp)

      features_importances = dict()
      model_feature_importances_ = model.feature_importances_

      for i in range(len(dataset.columns)-1):
        features_importances.update({dataset.columns[i+1]: model_feature_importances_[i]})
      features_importances = dict(sorted(features_importances.items(), key=lambda pair: pair[1], reverse=True))

      return features_importances

  X_time = []
  y_time = []

  print('loading...', end='\r')
  
  for i in range(1, 11, 2):
    now = time.time()
    _ = get_fnames(i)
    end = time.time()
    X_time.append([i])
    y_time.append([end-now])
    print('            ', end='\r')
    print('loading' + ''.join(['.']*(i//2%9)), end='\r')
  
  time_model = LinearRegression()
  time_model.fit(X_time, y_time)
  work_time = time_model.predict([[TreeNumber]])[0,0]
  
  now_st = time.gmtime(time.time())
  print('Время начало работы:\t{}:{}:{}'.format(now_st.tm_hour+3+2, now_st.tm_min, now_st.tm_sec))

  end_st = time.gmtime(time.time() + work_time)
  print('Примерный конец работы: {}:{}:{}'.format(end_st.tm_hour+3+2, end_st.tm_min, end_st.tm_sec))

  features_importances = get_fnames(TreeNumber)
  feat_df = pd.DataFrame({'feature': list(features_importances.keys()),
                          'value':   list(features_importances.values())})  

  now_st = time.gmtime(time.time())

  print('Время конца работы:\t{}:{}:{}\n'.format(now_st.tm_hour+3+2, now_st.tm_min, now_st.tm_sec))

  return feat_df





def B_search(dataset,
             target_name,
             TreeNumber):

  X_tmp = dataset.iloc[:,1:].values
  y_tmp = dataset.iloc[:,0].values
  model = RandomForestRegressor(TreeNumber, n_jobs=-1)
  boruta_feature_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1337, max_iter = 20, perc = 90)
  boruta_feature_selector.fit(X_tmp, y_tmp)

  X_filtered = boruta_feature_selector.transform(X_tmp)

  final_features = list()
  features = dataset.drop(target_name, axis=1).columns
  indexes = np.where(boruta_feature_selector.support_ == True)
  for x in np.nditer(indexes):
      final_features.append(features[x])

  #print('Важные признаки найденные с помощью Boruta:')
  #for i, final_fname in enumerate(final_features):
  #  print(final_fname)

  feat_df = pd.DataFrame({'feature': final_features})

  return feat_df

import pandas as pd

def process_dataset(dataset, targetName, **features_dict):

        def _get_dataset(dataset,
                     
                         acceleration,
                         dyn_levels,
                         water_cut,                         
                         all_debits,
                     
                         lag_max = 30, 
                         column_types = ['Обводненность', 'Динамический уровень', 'Приемистость воды'],
                         statistic_features = True,
                         calendar_features  = True):
    
            mode = {'Динамический уровень': False,\
                    'Приемистость воды': False,\
                    'Дебит жидкости': False,\
                    'Обводненность': False}
      
            for col_type in column_types:
                mode[col_type] = True
    
    
            features, data_prep = {}, dataset.copy()
    
            if mode['Дебит жидкости']:
                for name in dyn_levels:
                    features.update({"frate_" + name[-4:]: []})
        
            if mode['Динамический уровень']:
                for name in dyn_levels:
                    features.update({"dyn_" + name[-4:]: []})

            if mode['Приемистость воды']:
                for name in acceleration:
                    features.update({"acc_" + name[-4:]: []})  

            if mode['Обводненность']:
                for name in water_cut:
                    features.update({"wcut_" + name[-4:]: []})  


            if lag_max > 0:
                for lag_num in range(1,lag_max+1,1):
    
                    if mode['Дебит жидкости']:
                        for name in all_debits:
                            data_prep["frate_"+name[-4:] + "_lag_{}".format(lag_num)] = data_prep[name].shift(lag_num)
                            features["frate_"+name[-4:]].append("frate_"+name[-4:] + "_lag_{}".format(lag_num))
    
                    if mode['Динамический уровень']:
                        for name in dyn_levels:
                            data_prep["dyn_" + name[-4:] + "_lag_{}".format(lag_num)] = data_prep[name].shift(lag_num)
                            features["dyn_" + name[-4:]].append("dyn_" + name[-4:] + "_lag_{}".format(lag_num))
    
                    if mode['Обводненность']:
                        for name in water_cut:
                            data_prep["wcut_"+ name[-4:] + "_lag_{}".format(lag_num)] = data_prep[name].shift(lag_num)
                            features["wcut_"+ name[-4:]].append("wcut_"+ name[-4:] + "_lag_{}".format(lag_num))
    
                    if mode['Приемистость воды']:
                        for name in acceleration:
                            data_prep["acc_" + name[-4:] + "_lag_{}".format(lag_num)] = data_prep[name].shift(lag_num)
                            features["acc_" + name[-4:]].append("acc_" + name[-4:] + "_lag_{}".format(lag_num))

            if statistic_features:
                for name, val in features.items():
                    data_prep[name+'_mean'] = data_prep[val].mean(axis=1)
                    data_prep[name+'_std'] = data_prep[val].std(axis=1)
                    data_prep[name+'_min']   = data_prep[val].min(axis=1)
                    data_prep[name+'_Q_0.1'] = np.quantile(data_prep[val], 0.1, axis=1)
                    data_prep[name+'_Q_0.2'] = np.quantile(data_prep[val], 0.2, axis=1)
                    data_prep[name+'_Q_0.3'] = np.quantile(data_prep[val], 0.3, axis=1)
                    data_prep[name+'_Q_0.4'] = np.quantile(data_prep[val], 0.4, axis=1)
                    data_prep[name+'_med']   = np.median(data_prep[val], axis=1)
                    data_prep[name+'_Q_0.6'] = np.quantile(data_prep[val], 0.6, axis=1)
                    data_prep[name+'_Q_0.7'] = np.quantile(data_prep[val], 0.7, axis=1)
                    data_prep[name+'_Q_0.8'] = np.quantile(data_prep[val], 0.8, axis=1)
                    data_prep[name+'_Q_0.9'] = np.quantile(data_prep[val], 0.9, axis=1)
                    data_prep[name+'_max']   = data_prep[val].max(axis=1)
                    data_prep[name+'_var'] = data_prep[val].var(axis=1)
    

            if calendar_features:
    
                def cal_type_features(cur_df, col_name):
    
                    df_tmp = pd.get_dummies(cur_df[col_name])
                    df_tmp.columns = list(map(lambda x: col_name + '_' + str(x), df_tmp.columns))
                    df_tmp.index = cur_df.index
          
                    return df_tmp
          
                df_tmp = pd.DataFrame()
                df_tmp['day'] = data_prep.index.day  
                df_tmp.index = data_prep.index

                # for col_name in ['month', 'week', 'dow', 'day']:
                for col_name in ['day']:
                    data_prep = pd.concat((data_prep, cal_type_features(df_tmp, col_name)), axis=1)
    
            return data_prep.dropna()



        def _smooth_dataset(dataset):
    
            window_size = 5 # Гиперпараметр.
            dataset.fillna(method='bfill', inplace=True) # Заполняет крайним справа
            dataset = dataset.rolling(window=window_size,
                                      min_periods=window_size).mean() # Сглаживаем ряд.
            return dataset  

    
# again function:
        kwargs = {'acceleration': list(filter(lambda x: "Приемистость"         in x, dataset.columns)),
                  'dyn_levels':   list(filter(lambda x: 'Динамический уровень' in x, dataset.columns)),
                  'water_cut':    list(filter(lambda x: 'Обводненность'        in x, dataset.columns)),
                  'all_debits':   list(filter(lambda x: 'Дебит'                in x, dataset.columns))}

        nick = ' '.join(targetName.split()[:-1])
        target_name = {'name': targetName, 'nick': nick}
        notTargets = nick

        column_names = [targetName]
        column_names.extend(list(filter(lambda x: notTargets not in x, dataset.columns)))

        targeted_dataset = dataset.loc[:,column_names]
        targeted_dataset.rename({targetName: nick}, axis=1, inplace=True)

        targeted_dataset = _smooth_dataset(targeted_dataset)

        all_column_types = ['Обводненность',\
                            'Динамический уровень',\
                            'Приемистость воды']

        kwargs.update(features_dict)

        final_dataset = _get_dataset(targeted_dataset, **kwargs)

        return final_dataset



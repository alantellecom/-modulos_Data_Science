
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from itertools import combinations

def val_couts_cols (Dataframe,cols):
  for x in cols:
    print('coluna: {0}, categorias: {1}'.format(x,len(Dataframe[x].value_counts())))
  print('Total Samples :' + str(len(Dataframe)))
  
def get_col_type(df,col_type):
  cols_types=df.dtypes.reset_index()
  cols_types.columns=['col','type']
  cols_type = cols_types.apply(lambda x: x['col'] if x['type']==col_type else np.nan ,axis=1)
  return cols_type.dropna()

def to_type(DataFrame, columns, type):
  DataFrame_aux = DataFrame.copy()
  for col in columns:
      DataFrame_aux[col]=DataFrame_aux[col].astype(type)
  return DataFrame_aux

def remove_incoherence(DataFrame,expression, replace_val, columns=[]):
  if len(columns)==0:
    columns = DataFrame.columns
    
  if str(replace_val) == str(np.nan):
    DataFrame_aux=DataFrame.replace(expression, replace_val, regex=True) # não usar str.replace pois não aceita np.nan
    return DataFrame_aux
  else: 
    for col in columns:
      i=0
      while (True): # quando trabalhamos com grupos no regex, ele não é capaz de substituir todos os grupos, então é necessario iterar a cada nova substituição
        DataFrame_aux[col]=DataFrame[col].str.replace(expression, replace_val, regex=True)
        #warnings.filterwarnings('ignore','UserWarning') # para evitar warning quando str.contains chamar expressions contendo groups que não serão utilizados
        num_matchs = len(DataFrame_aux[DataFrame_aux[col].str.contains(expression, na=False)])#  verifica se regex funcionou, caso sim retorna 0, senão retorna o numero de matchs
        DataFrame = DataFrame_aux
        
        if num_matchs == 0:
            break
        if i == 100:
            DataFrame_aux =pd.DataFrame([])
            break
        i+=1
    return DataFrame_aux
 
def group_low_freq_cats(DataFrame, col_name, threshold=0.01, name='others'):
  df = DataFrame.copy()
  cat_freq = df[col_name].value_counts()
  cat_low_freq = cat_freq[cat_freq/cat_freq.sum() <= threshold].index
  df.loc[df[col_name].isin(cat_low_freq),col_name]='others'
  return df

def feature_selection(Dataset, feature, target ,in_out, method='na'): 
  fs_score =[]
  oe = OrdinalEncoder()

  X = (np.array(Dataset[feature])).reshape(-1,1)
  oe.fit(X)
  X_enc = oe.transform(X)

  y = np.array(Dataset[target]).reshape(-1,1)
  oe.fit(y)
  y_enc = oe.transform(y)
  
  if in_out == 'cat_cat': 
    if method == 'chi2':
      fs = SelectKBest(score_func=chi2, k='all') 
    else:
      fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_enc, y_enc)
    fs_score = fs.scores_
  elif in_out == 'num_num':
    fs = SelectKBest(score_func=f_regression, k='all')
    fs.fit(X, y.ravel())
    fs_score = fs.scores_
  elif in_out == 'num_cat':
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X, y_enc)
    fs_score = fs.scores_
  elif in_out == 'cat_num':
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X_enc, y.ravel())
    fs_score = fs.scores_
  else:
    fs_score=[]

  return fs_score

def exclui_outliers(DataFrame, col_name):
  Q1 = DataFrame[col_name].quantile(.25)
  Q3 = DataFrame[col_name].quantile(.75)
  IIQ =Q3 -Q1
  limite_inf = Q1 -1.5*IIQ
  limite_sup = Q3 +1.5*IIQ
  
  return DataFrame[(DataFrame[col_name]>=limite_inf) & (DataFrame[col_name]<=limite_sup)]

def subplot_strip(Dataset,features,target):
  perm_features = list(combinations(features, 2))
  fig, axes = plt.subplots(len(perm_features),1,figsize=(10,len(perm_features)*10))
  
  for i, perm in enumerate(perm_features):
        sns.stripplot(ax=axes[i],data=Dataset,x=perm[0],y=perm[1], hue=target)
  plt.tight_layout(pad=3)

def boxplot_by_col(df,cat_cols,target):
  fig, ax = plt.subplots(len(cat_cols), 1, figsize=(25, 18))
  fig.subplots_adjust()
  t=0
  for var, subplot in zip(cat_cols, ax.flatten()):
      ax[t].set_xlabel(var,fontsize=18)
      sort_qtl_index = df.groupby(var)[target].quantile(0.5).sort_values().index
      sort_qtl_values = df.groupby(var)[target].quantile(0.5).sort_values()
      sns.boxplot(x=var, y=target, data=df, ax=subplot,order=sort_qtl_index)
      sns.pointplot(x=sort_qtl_index,y= sort_qtl_values,ax=subplot,color='r')
      t+=1    
  plt.tight_layout(pad=3)
  
def plot_hists_scatters(*args,cols=['none'],type_plot='scatter',target=[]):
  
  if np.array_equal(target,[]) & (type_plot  == 'scatter'):
    print('No target')
  elif len(args)==1:
    if type_plot  == 'scatter':
      plt.title(cols[0],fontsize=18)
      sns.scatterplot(x=args[0],y=target)
    else:
      plt.title(cols[0],fontsize=18)
      sns.histplot(args[0])
  else:
    fig, ax = plt.subplots(1, len(args), figsize=(10, 4))
    t=0
    for arg, subplot in zip(args,ax.flatten()):  
      if type_plot == 'hist':
        if len(cols) == 1:
          ax[t].set_title(cols[0],fontsize=18)
        else:
          ax[t].set_title(cols[t],fontsize=18)
        sns.histplot(arg, ax=subplot)
      else:
        if len(cols) == 1:
          ax[t].set_title(cols[0],fontsize=18)
        else:
          ax[t].set_title(cols[t],fontsize=18)
        sns.scatterplot(x=arg,y=target, ax=subplot)
      t+=1
    plt.tight_layout(pad=3)
    
def is_not_date(x): # using with df[df['col'].map(is_not_date)]
    import re
    return (len(re.findall('[0-9]+/[0-9]+/[0-9]+',x)))==0
 
def get_related_items(df_1):
    if df_1['col_filter'] == val_1:
        return  df_1['col_target']
    elif  df_1['col_filter'] == val_2: 
        return  df_2[df_2['col_id']==df_1['col_id']]['col_target'].values[0]
    else: # val_3
        return  df_3[df_3['col_id']==x['col_id']]['col_target'].values[0]
 
# break lines with datetime interval in discret lines by time unit (days, weeks, months, hours, etc)

df_souce['delta'] = df_souce['col_end'] - df_souce['col_start']
df_souce['delta'] = df_souce['delta'].apply(lambda x: (x.days + 1)) # days ou seconds (if other time units, we need to calculate the conversion)
df_break = df_source.copy()
for i, x in df_source.iterrows():
    if x['delta'] > 1:
        aux = pd.DataFrame([x]*x['delta'])
        days = pd.date_range(x['col_start'],x['col_end'],freq='D')
        aux['col_start'], aux['col_end'], aux['col_dur'] = days, days, 1    
        df_break= df_break.append(aux,ignore_index=True)
        
df_break.drop(df_break[df_break['delta']>1].index,inplace=True) # drop consolidate line

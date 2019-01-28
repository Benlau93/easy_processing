import numpy as np
import pandas as pd
from IPython.display import display


def display_all(df):
    """
    Set max rows and columns to display
    """
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            display(df)

def drop_missing(df,threshold=0.5,drop_cols=[]):
    """
    Process missing columns
    
    Parameters
    ----------
    df : 
    DataFrame
    
    threshold : default=0.5
    amount of missing value in columns required to drop the column
    
    drop_cols : default=[]
    list of columns to be dropped. If not given, function will drop column based on amount of missing values
    
    Returns
    ----------
    Dataframe with the columns dropped
    Dropped columns name as a list
    
    """
    
    if not drop_cols:
        rows = len(df)
        num_nonna = round((1-threshold) * rows,0)
        for k,v in (df.isnull().sum()/rows).items():
            if v>threshold:
                drop_cols.append(k)
        
        d= df.dropna(axis=1,thresh = num_nonna)
    else:
        d= df.drop(drop_cols,axis=1)
            
    
    return d,drop_cols



def proc_date(df,col,attr=["year","month","day"],drop=True):
    """
    Process datatime column
    
    Parameters
    ----------
    df : 
    DataFrame
    
    col : 
    single column name that contain date information
    
    attr : default= ["year","month","day"]
    attribute you wish to extract from the datetime column
    
    drop : default=True
    if True, drop the datetime column after processing
    
    
    """
    if not np.issubdtype(df[col],np.datetime64):
        df[col] = pd.to_datetime(df[col],infer_datetime_format=True)
    
    for ea in attr:
        df[col+"_"+ea] = getattr(df[col].dt,ea)
    
    if drop:
        df.drop(col,axis=1, inplace=True)


def fill_numeric(df,missing_val):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtypes):
            if df[col].isnull().sum():
                df[col].fillna(df[col].median(),inplace=True)
                missing_val[col] = df[col].median()
    return missing_val

def proc_miss(df,missing_val={}):
    """
    Process missing values
    
    Parameters
    ----------
    df : 
    DataFrame
    
    missing_val : default={}
    Dict with column names as keys, value to replace as values. If not given, function will replace numeric missing
    values with median of the respective column
    
    Returns
    ----------
    Dataframe with missing values filled 
    missing_val with columns as key and median of the respective column as values
    
    """
    d= df.copy()
    if not missing_val:
        missing_val = fill_numeric(d,missing_val)
    
    else:
        for k,v in missing_val.items():
            if d[k].isnull().sum():
                d[k].fillna(v,inplace=True)
        
        if d.isnull().sum().sum():
            for col in d.columns:
                missing_val = fill_numeric(d,missing_val)

    return d,missing_val


def convert_cat(df,cat_cols=[]):
    if cat_cols:
        for col in cat_cols:
                df[col] = df[col].astype("category")
    else:
        obj_columns = df.select_dtypes(['object']).columns
        for obj in obj_columns:
            df[obj] = df[obj].astype('category')
            cat_cols.append(obj)
    return df, cat_cols



def set_cat(df,cat_dict={}):
    if cat_dict:
        for k,v in cat_dict.items():
            df[k] = df[k].cat.set_categories(v)
    else:
        for col in df.columns:
            if df[col].dtypes.name =="category":
                cat_dict[col] = df[col].cat.categories
    return cat_dict

def gen_dummies(df,cat_cols,max_cardi):
    cardi_cols = []
    for col in cat_cols:
        if len(df[col].cat.categories) <= max_cardi:
            cardi_cols.append(col)
    
    df = pd.get_dummies(df,columns = cardi_cols,prefix=cardi_cols,drop_first=True)
    
    return df, cardi_cols


def cat_codes(df,cat_cols):
    for col in cat_cols:
        df[col] = df[col].cat.codes+1
    

def proc_cat(df,cat_cols=[],cat_dict={},max_cardi=None):
    """
    Process categorical variables
    
    Parameters
    ----------
    df: 
    DataFrame
    
    cat_cols : default=[]
    list of pre-determined categorical variables
    
    cat_dict : default={}
    Dict with categorical variables as keys and pandas.Series.cat.categories as values. If not given, cat_dict is
    generated with for every categorical columns
    
    max_cardi : default=None
    maximum cardinality of the categorical variables. Which is the number of class in the categorical features.
    Categories variables with cardinality less or equal to max_cardi will be onehotencoded to produce dummies variables
    
    
    Returns
    ----------
    Dataframe with categorical variables processed
    cat_dict with categorical columns as key and respective pandas.Series.cat.categories as values
    
    """
    d = df.copy()
    
    d, cat_cols = convert_cat(d,cat_cols)

    cat_dict = set_cat(d,cat_dict)
    
    if max_cardi:
        d,cardi_cols = gen_dummies(d,cat_cols,max_cardi)
        cat_cols = list(set(cat_cols) - set(cardi_cols))
    
    cat_codes(d,cat_cols)
    
    return d, cat_dict

def train_valid_split(df,num_valid,shuffle=False):
    """
    Split df into training and validation set
    
    Parameters
    ----------
    df : 
    DataFrame
    
    num_valid : 
    number of samples needed in validation set
    
    shuffle : default=False
    Shuffle the rows to randomly sample training and validation sets
    
    Returns
    ----------
    Training and validation set respectively
    
    """
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    n_trn = len(df) - num_valid
    n_train = df[:n_trn]
    n_valid = df[n_trn:]
    
    return n_train, n_valid

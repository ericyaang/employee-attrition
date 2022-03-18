# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
seed = 123

# Read data
df = pd.read_csv(
    r'C:\Users\Eric\Documents\employee-attrition\employee-attrition\data\employee_churn_data.csv')

# Set test set aside
train, test = train_test_split(
    df, test_size=0.2, random_state=seed, stratify=df["left"])

# Enconding target variable
train["left"] = train["left"].map({'yes': 1, 'no': 0})
test["left"] = test["left"].map({'yes': 1, 'no': 0})

# Encoding tenure as integer
train["tenure"] = train["tenure"].astype("int8")
test["tenure"] = test["tenure"].astype("int8")

# Encoding categoricals

train["salary"] = train["salary"].astype("category")
test["salary"] = test["salary"].astype("category")
train["department"] = train["department"].astype("category")
test["department"] = test["department"].astype("category")

# Encode correcty labels for each variable


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8', 'int16', 'int32',
                'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# Save them as pickles
train.to_pickle('train_enc.pkl')
test.to_pickle('test_enc.pkl')

import numpy as np
from pandas import DataFrame


def is_outlier(x):

    IQR = np.percentile(x, 75) - np.percentile(x, 25)
    upper_fence = np.percentile(x, 75) + (IQR * 1.5)
    lower_fence = np.percentile(x, 25) - (IQR * 1.5)

    return (x > upper_fence) | (x < lower_fence)


def clean(df: DataFrame) -> DataFrame:

    group_columns = df.columns.to_list()
    group_columns.remove("inference_time")

    df = df[~df.groupby(group_columns)["inference_time"].apply(is_outlier)]

    return df

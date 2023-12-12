import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import clean

def main():
    df = pd.read_csv("batch_scaling.csv", header=0, sep=",")
    df.drop(columns=["copy"])
    
    df.loc[:, ["inference_time"]] = df.loc[:, ["inference_time"]] / 10 ** 9

    df = clean(df)

    sns.catplot(data=df, y="inference_time", hue="threads", x="batch_size", col="sgx", row="nn", kind="bar",
                legend_out=True, sharey="row", width=0.5,
                estimator=np.min,
                # errorbar=("pi", 50), errwidth=1.5, capsize=0.1
                )

    plt.savefig(f"batch_scaling.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()
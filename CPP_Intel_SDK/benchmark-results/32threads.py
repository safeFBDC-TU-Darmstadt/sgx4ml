import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import clean

def main():
    df = pd.read_csv("32threads.csv", header=0, sep=",")
    df2 = pd.read_csv("rerun_2023_scalable.csv", header=0, sep=",")
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    df.drop(columns=["copy"])
    # df = df[df["batch_size"] == 1]

    df.loc[:, ["inference_time"]] = df.loc[:, ["inference_time"]] / 10 ** 9
    df = clean(df)

    sns.catplot(data=df, y="inference_time", hue="threads", x="batch_size", col="sgx", row="nn", kind="bar",
                legend_out=True, sharey="row", width=0.5,
                estimator=np.median,
                errorbar=("pi", 50), errwidth=1.5, capsize=0.05
                )

    plt.savefig(f"32threads.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()
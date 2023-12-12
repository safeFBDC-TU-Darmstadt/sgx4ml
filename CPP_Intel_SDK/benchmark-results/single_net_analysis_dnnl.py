import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from helper_functions import clean


def load_gramine_data(threads: int) -> DataFrame:
    gramine_values = pd.read_csv(f"PyTorch_Gramine_{threads}.csv", header=0, sep=",")
    gramine_values.columns = ["nn", "batch_size", "min", "max", "avg"]
    gramine_values["threads"] = threads
    gramine_values["sys"] = "pt"
    gramine_values["sgx"] = "yes"

    return gramine_values


def load_all_data() -> DataFrame:
    dnnl_values = pd.read_csv("pytorch_comparison_ext.csv", header=0, sep=",")
    dnnl_values.loc[:, ["inference_time"]] = dnnl_values.loc[:, ["inference_time"]] / 10**6
    dnnl_values["sys"] = "dnnl"

    dnnl_values = clean(dnnl_values)

    return dnnl_values


def speedup_comparison(df: DataFrame, net: str, batch_size: int):
    df = df[(df["nn"] == net) & (df["batch_size"] == batch_size)]

    sns.catplot(data=df, x="sgx", y="inference_time", hue="threads", kind="bar", legend_out=True)
    plt.title(f"Inference time for {net.upper()}. Batch size = {batch_size}")

    plt.savefig(f"{net}_dnnl_speedup_comp_b{batch_size}.pdf", bbox_inches='tight')


def sgx_comparison(df: DataFrame, net: str, batch_size: int):
    df = df[(df["nn"] == net) & (df["batch_size"] == batch_size)]

    sns.catplot(data=df, x="threads", y="inference_time", hue="sgx", kind="bar", legend_out=True)
    plt.title(f"Inference time for {net.upper()}. Batch size = {batch_size}")

    plt.savefig(f"{net}_dnnl_sgx_comp_b{batch_size}.pdf", bbox_inches='tight')


def speedup_comparison_batch_subplots(df: DataFrame, net: str, kind: str = "bar"):
    df = df[(df["nn"] == net)]

    sns.catplot(data=df, x="sgx", y="inference_time", hue="threads", col="batch_size", kind="bar", legend_out=True,
                sharey=True)
    # plt.title(f"Inference time for {net.upper()}")

    plt.savefig(f"{net}_dnnl_speedup_comp_b_all.pdf", bbox_inches='tight')


def sgx_comparison_batch_subplots(df: DataFrame, net: str):
    df = df[(df["nn"] == net)]

    sns.catplot(data=df, x="threads", y="inference_time", hue="sgx", col="batch_size", kind="bar", legend_out=True,
                sharey=True)
    # plt.title(f"Inference time for {net.upper()}")

    plt.savefig(f"{net}_dnnl_sgx_comp_b_all.pdf", bbox_inches='tight')


def main():
    df = load_all_data()

    for net in ["mlp", "cnn", "vgg19"]:
        speedup_comparison(df, net, 1)
        sgx_comparison(df, net, 1)
        speedup_comparison_batch_subplots(df, net)
        sgx_comparison_batch_subplots(df, net)


if __name__ == '__main__':
    main()

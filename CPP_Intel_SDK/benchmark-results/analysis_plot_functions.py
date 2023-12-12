import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import clean


NETWORK_ORDER = ["mlp", "cnn", "medium_mlp", "large_mlp", "simple_alex", "alex", "vgg16", "vgg19"]


def load_pytorch_data() -> DataFrame:
    pytorch_values_scalable = pd.read_csv(f"pytorch_scalable_2.csv", header=0, sep=",")
    medium_large_mlp_scalable = pd.read_csv(f"medium_large_mlp_pytorch_scalable.csv", header=0, sep=",")
    pytorch_values_scalable = pd.concat([pytorch_values_scalable, medium_large_mlp_scalable], axis=0, ignore_index=True)
    pytorch_values_scalable["server"] = "scalable"

    pytorch_values_client = pd.read_csv(f"pytorch_client.csv", header=0, sep=",")
    medium_large_mlp_client = pd.read_csv(f"medium_large_mlp_pytorch_client.csv", header=0, sep=",")
    pytorch_values_client = pd.concat([pytorch_values_client, medium_large_mlp_client], axis=0, ignore_index=True)
    pytorch_values_client["server"] = "client"

    pytorch_values = pd.concat([pytorch_values_scalable, pytorch_values_client], axis=0, ignore_index=True)
    pytorch_values.loc[pytorch_values["nn"] == "large_mlp1", ["nn"]] = "medium_mlp"
    pytorch_values.loc[pytorch_values["nn"] == "large_mlp2", ["nn"]] = "large_mlp"
    pytorch_values["copy"] = "yes"
    pytorch_values["sys"] = "pt"
    pytorch_values.loc[:, ["inference_time"]] = pytorch_values.loc[:, ["inference_time"]] / 10 ** 9

    return pytorch_values


def load_all_data() -> DataFrame:
    dnnl_values_scalable = pd.read_csv("rerun_2023_scalable.csv", header=0, sep=",")
    # vgg19_scalable = pd.read_csv("BTW-Paper/freq_fixed_scalable_vgg19.csv", header=0, sep=",")
    # alex_scalable = pd.read_csv("BTW-Paper/alexnet_scalable.csv", header=0, sep=",")
    # dnnl_values_scalable = pd.concat([dnnl_values_scalable, vgg19_scalable, alex_scalable], axis=0, ignore_index=True)
    dnnl_values_scalable["server"] = "scalable"

    dnnl_values_client = pd.read_csv("rerun_2023_client.csv", header=0, sep=",")
    # vgg19_client = pd.read_csv("BTW-Paper/freq_fixed_client_vgg19.csv", header=0, sep=",")
    # alex_client = pd.read_csv("BTW-Paper/alexnet_client.csv", header=0, sep=",")
    # dnnl_values_client = pd.concat([dnnl_values_client, vgg19_client, alex_client], axis=0, ignore_index=True)
    dnnl_values_client["server"] = "client"

    all_dnnl_data = pd.concat([dnnl_values_scalable, dnnl_values_client], axis=0, ignore_index=True)

    all_dnnl_data.loc[:, ["inference_time"]] = all_dnnl_data.loc[:, ["inference_time"]] / 10 ** 9
    all_dnnl_data["sys"] = "dnnl"
    all_dnnl_data = clean(all_dnnl_data)

    all_data = pd.concat([all_dnnl_data, load_pytorch_data()])

    return all_data


def copy_comparison(df: DataFrame):
    df = df[(df["sys"] == "dnnl")
            & (df["sgx"] == "yes")
            & (df["batch_size"] == 1)
        # & (df["nn"] == "large_mlp")
        # & (df["server"] == "scalable")
            ]

    sns.catplot(data=df, x="threads", y="inference_time", hue="copy", col="server", row="nn", kind="bar",
                legend_out=True, sharey="row")

    plt.savefig(f"dnnl_copy_comp.pdf", bbox_inches='tight')


def server_comparison(df: DataFrame, sgx: bool):
    df = df[(df["sys"] == "dnnl")
            & (((df["sgx"] == "yes") & (df["copy"] == "yes")) if sgx else (df["sgx"] == "no"))
            & (df["batch_size"] == 1)
        # & (df["threads"] == 8)
        # & (df["nn"] == "vgg19")
        # & (df["server"] == "scalable")
            ]

    sns.catplot(data=df, y="inference_time", hue="threads", x="server", col="nn",
                col_order=NETWORK_ORDER,
                kind="bar", sharey=False,
                legend_out=True, width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"dnnl_server_comp_{'sgx' if sgx else 'native'}.pdf", bbox_inches='tight')


def server_comparison_vgg(df: DataFrame):
    df = df[(df["sys"] == "dnnl")
            & (((df["sgx"] == "yes") & (df["copy"] == "yes")) | (df["sgx"] == "no"))
            & (df["batch_size"] == 1)
            # & (df["threads"] == 8)
            & (df["nn"] == "vgg19")
        # & (df["server"] == "scalable")
            ]

    sns.catplot(data=df, y="inference_time", hue="threads", x="server", col="sgx", kind="bar", legend_out=True,
                sharey=True, log=False)

    plt.savefig(f"dnnl_server_comp_vgg19.pdf", bbox_inches='tight')


def system_comparison(df: DataFrame, server: str):
    df = df[(((df["sgx"] == "yes") & (df["copy"] == "yes")) | (df["sgx"] == "no"))
            & (df["batch_size"] == 1)
            # & (df["threads"] == 8)
            # & (df["nn"].isin(["vgg19", "cnn"]))
            & (df["server"] == server)
            ]

    sns.catplot(data=df, y="inference_time", hue="threads", x="sys", col="sgx", row="nn",
                row_order=NETWORK_ORDER, kind="bar", legend_out=True, sharey="row",
                width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"system_comp_{server}.pdf", bbox_inches='tight')


def batch_size_comparison(df: DataFrame, server: str):
    df = df[(df["sys"] == "dnnl")
            & (((df["sgx"] == "yes") & (df["copy"] == "yes")) | (df["sgx"] == "no"))
            # & (df["batch_size"] == 1)
            # & (df["threads"] == 8)
            # & (df["nn"].isin(["vgg19", "cnn"]))
            & (df["server"] == server)
            ]

    sns.catplot(data=df, y="inference_time", hue="threads", x="batch_size", col="sgx", row="nn",
                row_order=NETWORK_ORDER, kind="bar", legend_out=True, sharey="row",
                width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"batch_size_comp_{server}.pdf", bbox_inches='tight')


def sgx_comparison(df: DataFrame):
    df = df[((df["sgx"] == "yes") & (df["copy"] == "yes") | (df["sgx"] == "no"))
            & (df["batch_size"] == 1)
            & (df["sys"] == "dnnl")
            # & (df["threads"] == 8)
            # & (df["nn"].isin(["vgg19", "cnn"]))
            # & (df["server"] == "scalable")
            ]

    sns.catplot(data=df, y="inference_time", hue="sgx", x="threads", col="server", row="nn",
                row_order=NETWORK_ORDER, kind="bar", legend_out=True, sharey="row",
                width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"sgx_comp_all.pdf", bbox_inches='tight')


def single_net(df: DataFrame, net: str):
    df = df[(df["nn"] == net)
            # & (df["sgx"] == "no")
            & ((df["sgx"] == "yes") & (df["copy"] == "yes") | (df["sgx"] == "no"))
            & (df["batch_size"] == 1)
        # & (df["threads"] == 8)
        # & (df["sys"] == "dnnl")
        # & (df["server"] == "scalable")
            ]

    sns.catplot(data=df, y="inference_time", hue="threads", x="sys", col="sgx", row="server", kind="bar",
                legend_out=True, sharey=True, width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"single_{net}.pdf", bbox_inches='tight')


def sgx_comparison_single(df: DataFrame, net: str):
    df = df[(df["nn"] == net)
            # & (df["sys"] == "dnnl")
            & (df["batch_size"] == 1)
            & ((df["sgx"] == "yes") & (df["copy"] == "yes") | (df["sgx"] == "no"))
            ]

    sns.catplot(data=df, y="inference_time", hue="sgx", x="threads", col="sys", row="server", kind="bar",
                legend_out=True, sharey=True, width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"sgx_comp_{net}.pdf", bbox_inches='tight')


def baselines(df: DataFrame, sys: str, batch_size: int):
    df = df[(df["sys"] == sys)
            & (df["batch_size"] == batch_size)
            & (df["sgx"] == "no")
            ]

    sns.catplot(data=df, y="inference_time", x="threads", col="nn", row="server", kind="bar",
                col_order=NETWORK_ORDER,
                legend_out=True, sharey=False, width=0.5,
                estimator=np.median, errorbar=("pi", 50), errwidth=1.5, capsize=0.1)

    plt.savefig(f"baselines_{sys}_b{batch_size}.pdf", bbox_inches='tight')


def main():
    df = load_all_data()
    print(df.columns)
    server_comparison(df, False)
    server_comparison(df, True)
    # server_comparison_vgg(df)
    # single_net(df, "mlp")
    # single_net(df, "cnn")
    # single_net(df, "vgg19")
    # single_net(df, "medium_mlp")
    # single_net(df, "large_mlp")
    # single_net(df, "alex")
    # system_comparison(df, "scalable")
    # system_comparison(df, "client")
    batch_size_comparison(df, "scalable")
    batch_size_comparison(df, "client")
    sgx_comparison(df)
    # sgx_comparison_single(df, "vgg19")
    # sgx_comparison_single(df, "mlp")
    # sgx_comparison_single(df, "cnn")
    # sgx_comparison_single(df, "medium_mlp")
    # sgx_comparison_single(df, "large_mlp")
    # sgx_comparison_single(df, "alex")
    baselines(df, "dnnl", 1)
    baselines(df, "dnnl", 16)


if __name__ == '__main__':
    main()

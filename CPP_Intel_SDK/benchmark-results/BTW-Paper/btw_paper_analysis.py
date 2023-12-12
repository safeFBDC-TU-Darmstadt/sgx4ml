import pandas as pd
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

matplotlib.rcParams['hatch.linewidth'] = 0.7

def is_outlier(x):

    IQR = np.percentile(x, 75) - np.percentile(x, 25)
    upper_fence = np.percentile(x, 75) + (IQR * 1.5)
    lower_fence = np.percentile(x, 25) - (IQR * 1.5)

    return (x > upper_fence) | (x < lower_fence)


def clean(df: DataFrame) -> DataFrame:

    group_columns = df.columns.to_list()
    group_columns.remove("inference_time")

    df = df[~df.groupby(group_columns, group_keys=False)["inference_time"].apply(is_outlier)]

    return df


def load_all_data() -> DataFrame:
    dnnl_values_scalable = pd.read_csv("../freq_fixed_2023_scalable.csv", header=0, sep=",")
    # vgg19_scalable = pd.read_csv("freq_fixed_scalable_vgg19.csv", header=0, sep=",")
    # alex_scalable = pd.read_csv("alexnet_scalable.csv", header=0, sep=",")
    # dnnl_values_scalable = pd.concat([dnnl_values_scalable, vgg19_scalable, alex_scalable], axis=0, ignore_index=True)
    dnnl_values_scalable["server"] = "scalable"

    dnnl_values_client = pd.read_csv("../freq_fixed_2023_client.csv", header=0, sep=",")
    # vgg19_client = pd.read_csv("freq_fixed_client_vgg19.csv", header=0, sep=",")
    # alex_client = pd.read_csv("alexnet_client.csv", header=0, sep=",")
    # dnnl_values_client = pd.concat([dnnl_values_client, vgg19_client, alex_client], axis=0, ignore_index=True)
    dnnl_values_client["server"] = "client"

    # dnnl_values_client = dnnl_values_client.loc[dnnl_values_client["nn"] != "alex", :]
    # dnnl_values_client.drop(dnnl_values_client["nn"] == "alex", inplace=True)
    # dnnl_values_client.loc[dnnl_values_client["nn"] == "simple_alex", "nn"] = "alex"

    all_dnnl_data = pd.concat([dnnl_values_scalable, dnnl_values_client], axis=0, ignore_index=True)

    all_dnnl_data = all_dnnl_data.loc[all_dnnl_data["nn"] != "alex", :]
    # dnnl_values_client.drop(dnnl_values_client["nn"] == "alex", inplace=True)
    all_dnnl_data.loc[all_dnnl_data["nn"] == "simple_alex", "nn"] = "alex"

    all_dnnl_data.loc[:, ["inference_time"]] = all_dnnl_data.loc[:, ["inference_time"]] / 10 ** 9
    all_dnnl_data["sys"] = "dnnl"
    all_dnnl_data = clean(all_dnnl_data)

    return all_dnnl_data


def paper_figure_t_shirt(df: DataFrame):
    # def setup_string(row):
    #     translation = {"noscalable": "SGXv2 Server w/o SGX",
    #                    "noclient": "SGXv1 Server w/o SGX",
    #                    "yesscalable": "SGXv2 Server w/ SGX",
    #                    "yesclient": "SGXv1 Server w/ SGX"}
    #     return translation[row["sgx"] + row["server"]]
    #
    # df["setup"] = df.loc[:, ["sgx", "server"]].apply(setup_string, axis=1)
    df.loc[df["server"] == "client", ["server"]] = "SGXv1"
    df.loc[df["server"] == "scalable", ["server"]] = "SGXv2"

    df = df[(df["batch_size"] == 1)
            & (df["nn"].isin(["mlp", "cnn", "alex", "medium_mlp", "large_mlp", "vgg19"]))
            & ((df["sgx"] == "yes") & (df["copy"] == "yes") | (df["sgx"] == "no"))
            & (df["sys"] == "dnnl")
            & (df["threads"] == 1)
            ]

    df.loc[df["nn"] == "mlp", ["nn"]] = "Small MLP"
    df.loc[df["nn"] == "cnn", ["nn"]] = "CNN"
    df.loc[df["nn"] == "alex", ["nn"]] = "Simple AlexNet"
    df.loc[df["nn"] == "medium_mlp", ["nn"]] = "Medium MLP"
    df.loc[df["nn"] == "large_mlp", ["nn"]] = "Large MLP"
    df.loc[df["nn"] == "vgg19", ["nn"]] = "VGG19"

    df = df.drop(["batch_size", "threads", "copy", "sys"], axis=1)

    medians = df.groupby(["sgx", "server", "nn"])["inference_time"].mean()

    print(medians.to_string())

    normalized = medians["yes"] / medians["no"]
    normalized = normalized.reset_index()

    # normalized["inference_time"] = (normalized["inference_time"] - 1) * 100

    print(normalized.to_string())

    sns.set_style("whitegrid")

    plot = sns.catplot(data=normalized, y="inference_time", hue="server",
                       hue_order=["SGXv1", "SGXv2"],
                       x="nn",
                       order=["Small MLP", "Simple AlexNet", "VGG19"],
                       kind="bar",
                       # estimator=np.median, errorbar=("pi", 50),
                       # estimator=np.min, errorbar=None,
                       # errwidth=1.5, capsize=0.1,
                       palette="tab10", log=False, legend=False, height=2.7, aspect=2.0)

    plot.set_axis_labels("Neural Network", "Rel. Slowdown over Native")

    plt.legend(loc="upper left", title="Server")
    plt.yticks([1,2,3,4,5,6])

    plt.savefig(f"paper_figure_t_shirt.pdf")


def paper_figure_t_shirt_improved(df: DataFrame):
    # def setup_string(row):
    #     translation = {"noscalable": "SGXv2 Server w/o SGX",
    #                    "noclient": "SGXv1 Server w/o SGX",
    #                    "yesscalable": "SGXv2 Server w/ SGX",
    #                    "yesclient": "SGXv1 Server w/ SGX"}
    #     return translation[row["sgx"] + row["server"]]
    #
    # df["setup"] = df.loc[:, ["sgx", "server"]].apply(setup_string, axis=1)
    df.loc[df["server"] == "client", ["server"]] = "SGXv1"
    df.loc[df["server"] == "scalable", ["server"]] = "SGXv2"

    df = df[(df["batch_size"] == 1)
            & (df["nn"].isin(["mlp", "cnn", "alex", "medium_mlp", "large_mlp", "vgg19"]))
            & ((df["sgx"] == "yes") & (df["copy"] == "yes") | (df["sgx"] == "no"))
            & (df["sys"] == "dnnl")
            & (df["threads"] == 1)
            ]

    df.loc[df["nn"] == "mlp", ["nn"]] = "Small MLP"
    df.loc[df["nn"] == "cnn", ["nn"]] = "CNN"
    df.loc[df["nn"] == "alex", ["nn"]] = "Simple AlexNet"
    df.loc[df["nn"] == "medium_mlp", ["nn"]] = "Medium MLP"
    df.loc[df["nn"] == "large_mlp", ["nn"]] = "Large MLP"
    df.loc[df["nn"] == "vgg19", ["nn"]] = "VGG19"

    df = df.drop(["batch_size", "threads", "copy", "sys"], axis=1)

    medians = df.groupby(["sgx", "server", "nn"])["inference_time"].mean()

    print(medians.to_string())

    normalized = medians["yes"] / medians["no"]
    normalized = normalized.reset_index()

    # normalized["inference_time"] = (normalized["inference_time"] - 1) * 100

    print(normalized.to_string())

    sns.set_style("whitegrid")

    plot = sns.catplot(data=normalized, y="inference_time", hue="server",
                       hue_order=["SGXv1", "SGXv2"],
                       x="nn",
                       order=["Small MLP", "Simple AlexNet", "VGG19"],
                       kind="bar",
                       # estimator=np.median, errorbar=("pi", 50),
                       # estimator=np.min, errorbar=None,
                       # errwidth=1.5, capsize=0.1,
                       # palette="tab10",
                       log=False, legend=False, height=2.7, aspect=2.0)

    plot.set_axis_labels("Neural Network", "Rel. Slowdown over Native")

    for i, sgx in zip(plot.ax.containers, ["SGXv1", "SGXv2"]):
        nets = ['Small MLP', 'Simple AlexNet', 'VGG19']
        # plot.ax.bar_label(i, labels=[f"{medians[('yes', sgx, 'Small MLP')]*10**6:.1f}µs\n{medians[('no', sgx, 'Small MLP')]*10 ** 6:.1f}µs",
        #                              f"{medians[('yes', sgx, 'Simple AlexNet')]*10**3:.1f}ms\n{medians[('no', sgx, 'Simple AlexNet')]*10**3:.1f}ms",
        #                              f"{medians[('yes', sgx, 'VGG19')]:.2f}s\n{medians[('no', sgx, 'VGG19')]:.2f}s"])

        plot.ax.bar_label(i, labels=[f"{medians[('yes', sgx, 'Small MLP')]*10**6:.1f}µs",
                                     f"{medians[('yes', sgx, 'Simple AlexNet')]*10**3:.1f}ms",
                                     f"{medians[('yes', sgx, 'VGG19')]:.2f}s"])

    #for bar, hatch in zip(plot.ax.patches, ['//'] * 3 + ['\\\\'] * 3):
    #    bar.set_hatch(hatch)

    plt.legend(loc="upper left", title="Server")
    plt.yticks([1,2,3,4,5,6])
    # plt.ylim((0,6.5))

    plt.savefig(f"paper_figure_t_shirt_improved.pdf")


def paper_figure_scaling():
    df = pd.read_csv("freq_fixed_scalable_vgg19.csv", header=0, sep=",")

    df32 = pd.read_csv("freq_fixed_scalable_vgg19_32.csv", header=0, sep=",")
    df32 = df32[df32["threads"] == 32]
    # df32.loc[df32["threads"] == 16, ["threads"]] = 17
    # df32.loc[df32["threads"] == 8, ["threads"]] = 9
    df = pd.concat([df, df32], axis=0, ignore_index=True)

    df = df[df["batch_size"] == 1]

    df.loc[:, ["inference_time"]] = df.loc[:, ["inference_time"]] / 10 ** 9

    df.drop(["nn", "copy", "batch_size"], axis=1)

    mins = df.groupby(["sgx", "threads"])["inference_time"].median()

    only_sgx = mins["yes"]
    # only_sgx.index = only_sgx.index.droplevel(0)
    speed_ups = only_sgx[1] / only_sgx
    print("SpeedUps with SGX:")
    print(speed_ups.to_string())

    no_sgx = mins["no"]
    speed_ups = no_sgx[1] / no_sgx
    print("SpeedUps without SGX:")
    print(speed_ups.to_string())

    mins = mins.reset_index()

    mins.loc[mins["sgx"] == "no", ["sgx"]] = "Without SGXv2"
    mins.loc[mins["sgx"] == "yes", ["sgx"]] = "With SGXv2"

    sns.set_style("whitegrid")

    plot = sns.catplot(data=mins, y="inference_time", hue="sgx",
                       hue_order=["Without SGXv2", "With SGXv2"],
                       x="threads",
                       kind="bar",
                       # estimator=np.median, errorbar=("pi", 50),
                       # estimator=np.min, errorbar=None,
                       # errwidth=1.5, capsize=0.1,
                       palette=sns.color_palette("tab10")[2:], log=False, legend=False, height=2.7, aspect=2.0)

    plot.set_axis_labels("Number of Threads", "Inference Time for VGG19 in s")

    plt.legend(loc="upper right", title="SGX")

    plt.savefig(f"paper_figure_scaling.pdf")


def paper_figure_real_scaling():
    df = pd.read_csv("freq_fixed_scalable_vgg19.csv", header=0, sep=",")

    df32 = pd.read_csv("freq_fixed_scalable_vgg19_32.csv", header=0, sep=",")
    df32 = df32[df32["threads"] == 32]
    # df32.loc[df32["threads"] == 16, ["threads"]] = 17
    # df32.loc[df32["threads"] == 8, ["threads"]] = 9
    df = pd.concat([df, df32], axis=0, ignore_index=True)

    df = df[df["batch_size"] == 1]

    # df.loc[:, ["inference_time"]] = df.loc[:, ["inference_time"]] / 10 ** 9

    df.drop(["nn", "copy", "batch_size"], axis=1)

    medians = df.groupby(["sgx", "threads"])["inference_time"].median()

    only_sgx = medians["yes"]
    # only_sgx.index = only_sgx.index.droplevel(0)
    speed_ups = only_sgx[1] / only_sgx
    print("SpeedUps with SGX:")
    print(speed_ups.to_string())

    no_sgx = medians["no"]
    speed_ups = no_sgx[1] / no_sgx
    print("SpeedUps without SGX:")
    print(speed_ups.to_string())

    # medians = medians.reset_index()

    medians = df.groupby(["threads", "sgx"])["inference_time"].median()
    all_speed_ups = medians[1] / medians
    all_speed_ups = all_speed_ups.reset_index()

    print(all_speed_ups.to_string())

    all_speed_ups.loc[all_speed_ups["sgx"] == "no", ["sgx"]] = "Without SGXv2"
    all_speed_ups.loc[all_speed_ups["sgx"] == "yes", ["sgx"]] = "With SGXv2"

    sns.set_style("whitegrid")

    plot = sns.catplot(data=all_speed_ups, y="inference_time", hue="sgx",
                       hue_order=["Without SGXv2", "With SGXv2"],
                       x="threads",
                       kind="bar",
                       # estimator=np.median, errorbar=("pi", 50),
                       # estimator=np.min, errorbar=None,
                       # errwidth=1.5, capsize=0.1,
                       palette=sns.color_palette()[2:], log=False, legend=False, height=2.7, aspect=2.0)

    plot.set_axis_labels("Number of Threads", "Speedup for VGG19")

    # plt.yscale('log', base=2)

    for i in plot.ax.containers:
        plot.ax.bar_label(i, fmt="%.1f")

    for bar, hatch in zip(plot.ax.patches, ['//'] * 6 + ['\\\\'] * 6):
        bar.set_hatch(hatch)

    plt.legend(loc="upper left", title="SGX")
    plt.savefig(f"paper_figure_real_scaling.pdf")


def main():
    df = load_all_data()
    print(df.columns)
    paper_figure_t_shirt_improved(df.copy())
    #paper_figure_real_scaling()


if __name__ == '__main__':
    main()

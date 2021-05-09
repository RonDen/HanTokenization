#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.04.19

""" Draw the result curves. """

import matplotlib.pyplot as plt
import argparse

def get_data(source_path, data_type, epochs_num, report_num):
    data = [0.] * epochs_num
    total_num = 0
    with open(source_path, "r", encoding="utf-8") as fr:
        for index, line in enumerate(fr):
            if "The 2-th fold as the validation set..." in line: break
            if data_type == "loss":
                if "Epoch id:" in line:
                    epoch_id = int(line.split(",")[0].split(":")[-1].strip())
                    data[epoch_id - 1] += float(line.split(",")[-1].split(":")[-1].strip())
            elif data_type == "f1":
                if "Start evaluation." in line: break
                if "precision" in line:
                    data[total_num] = float(line.split(",")[-1].strip())
                    total_num += 1
    if data_type == "loss":
        for idx in range(epochs_num):
            data[idx] /= report_num
    return data

def draw_curve(source_paths, colors, labels, target_path, data_type, epochs_num, report_num):
    fig, ax = plt.subplots()
    x_data = [i for i in range(1, epochs_num + 1)]
    for idx, source_path in enumerate(source_paths):
        y_data = get_data(source_path, data_type, epochs_num, report_num)
        print(y_data)
        assert len(x_data) == len(y_data)
        p = ax.plot(x_data, y_data, colors[idx], label=labels[idx])
    ax.legend()
    ax.set_xlabel("Iteration")
    if data_type == "loss":
        ax.set_ylabel("Train loss")
        ax.set_title("Train loss curve")
    elif data_type == "f1":
        ax.set_ylabel("F1 score")
        ax.set_title("Validation F1 score curve")
    plt.tight_layout()
    fig.savefig(target_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_type", 
        choices=["loss", "f1"],
        default="f1",
        help="What kind of data do you want to draw on the figure.")
    
    args = parser.parse_args()

    source_paths = [
        "../logs/bilstm_training.out", 
        "../logs/bilstmcrf_training.out",
        "../logs/bilstm_new_merge_training.out",
        "../logs/bilstm_new_merge_random_training.out",
        "../logs/transformer_new_merge_separate_training.out"
    ]
    colors = ["r-", "b-", "y-", "m-", "g-"]
    labels = ["1", "2", "3", "4", "5"]
    target_path = "../results/" + args.data_type + "_curves.pdf"
    epochs_num = 50
    report_num = 4

    draw_curve(source_paths, colors, labels, target_path, args.data_type, epochs_num, report_num)
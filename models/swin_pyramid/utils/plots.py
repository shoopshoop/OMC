import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')


def plot_train_val_eval(summary_dict, save_dir, save_name, config):
    # save_every = config["trainer"]["save_every"]
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           figsize=(9, 6), sharex=True)
    ax[0].plot(summary_dict["train_loss"], label="train loss")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    # ax[0].set_yscale("log")
    ax[0].legend()

    ax[1].plot(summary_dict["val_metric"], label="validation loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    
    save_dir = os.path.join(save_dir, "%s.png" % save_name)
    plt.legend()
    plt.savefig(save_dir)
    plt.close(fig)
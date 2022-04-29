import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use('Agg')

from utils.general import build_model
from datasets.build import get_dataloader, get_dataset

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

def visualize_heatmaps(ckp_path, save_dir, epoch, idx, config):
    checkpoint_path = os.path.join(ckp_path)
    ckp_content = torch.load(checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    model.load_state_dict(ckp_content["model_state_dict"])

    train_loader = get_dataloader(config, "train", shuffle=False)
    trainv_data = get_dataset(config, "trainv")

    model.eval()
    with torch.no_grad():
        num = len(train_loader)
        for iter, (inputs, labels) in enumerate(train_loader):
            if iter % 30 == 0: print("Iter %d (Total %d)" % (iter, num))
            inputs = inputs.to(device)
            pred_heatmaps = model(inputs)
            break

    i = idx
    iheatmaps = pred_heatmaps[i, :, :, :]
    ilabels = labels[i, :, :, :]
    iinputs, _ = trainv_data.__getitem__(i)

    height, width = trainv_data.bbox[i][2], trainv_data.bbox[i][3]

    img = cv2.resize(iinputs.permute(1,2,0).cpu().numpy(), [width, height])/255
    plt.imshow(img)
    plt.axis('off')
    figpath = save_dir+"img"+ str(i)+".jpg"
    plt.savefig(figpath, bbox_inches='tight')
    # plt.show()

    fig, ax = plt.subplots(4,9, figsize=(16,7))

    for il, (heatmap, label) in enumerate(zip(iheatmaps, ilabels)):
        row = il // 9
        col = il % 9

        new_heatmap = heatmap.cpu().numpy()
        ax[2*row+1, col].imshow(new_heatmap)
        ax[2*row+1, col].set_axis_off()

        new_label = label.cpu().numpy()
        ax[2*row, col].imshow(new_label)
        ax[2*row, col].set_axis_off()

    figpath = save_dir+"img"+str(i)+"_heatmaps_"+"epoch"+str(epoch)+".jpg"
    fig.patch.set_facecolor('black')
    plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
    plt.savefig(figpath, bbox_inches='tight')
    plt.close()
    # plt.show()
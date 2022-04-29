import json, os, argparse, time
import numpy as np
from numpy import append

import torch

from utils.train import train_iter, save_model_ckp, validate
from utils.general import get_optimizer, get_scheduler, build_model, set_device, seed_everything, get_loss_func, initialize_epoch_info, load_dict
from utils.plots import plot_train_val_eval
from configs.config import update_config, save_exp_info
from datasets.build import get_dataloader


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", required=True, type=str,
    #                     help="Path to the json config file.")
    parser.add_argument("--config", type=str,
                        default="C:/Users/jiang/Documents/AllLog/OMC/2022-Apr-27/16-06-28/Exp_Config.json",
                        help="Path to the json config file.")
    parser.add_argument("--machine", type=str,
                        default="pc")

    args = parser.parse_args()
    config = json.load(open(args.config, "r"))

    config = update_config(config, args)
    save_root, config = save_exp_info(config)

    fig_dir = os.path.join(save_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    mv_ckp_dir = os.path.join(save_root, "min_val_ckp")
    os.makedirs(mv_ckp_dir, exist_ok=True)

    seed_everything(config)
    device = set_device(config)

    model = build_model(config, device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    loss_func = get_loss_func(config)

    # Get DataLoader 
    train_loader = get_dataloader(config, mode="train")
    val_loader = get_dataloader(config, mode="val")

    num_train_iter_per_epoch = len(train_loader)
    print("Training Set Using %d samples" % (len(train_loader)*config["trainer"]["train_batch_size"]))
    print("Validation Set Using %d samples" % (len(val_loader)*config["trainer"]["test_batch_size"]))
    
    # Load Resume Training Configs
    if config["resume_training"]:
        checkpoint_path = os.path.join(config["ckp_dir"], "checkpoint.pth")
        cpk_content = torch.load(checkpoint_path)
        epoch = cpk_content["epoch"]
        total_iter = cpk_content["iter"]
        model_state_dict = cpk_content["model_state_dict"]
        model.load_state_dict(model_state_dict)
        opt_state_dict = cpk_content["optimizer_state_dict"]
        optimizer.load_state_dict(opt_state_dict)
        scheduler = cpk_content["scheduler"]
        print("CheckPoint Loaded")
        # === Create Dict to save Result
        summary = load_dict(config["ckp_dir"], name="summary.npy")
    else:
        epoch = 0
        total_iter = 0

        # === Create Dict to save Result
        summary = {  # To save training history
            "train_loss": [],
            "val_metric": [],
            "epoch_info": [],
            "min_val_metric": 0,
            }

    # Auto Data Parallel depending on gpu count
    if config["data_parallel"]:
        print("  >>> Multiple GPU Exsits. Use Data Parallel Training.")
        model = torch.nn.DataParallel(model)

    # ==== Start Training ====
    time_start = time.time()
    while epoch < config["trainer"]["num_epoch"]:
        print("Epoch [%d] Starts Training" % (epoch))

        epoch_info = initialize_epoch_info(config)
        iter_time_start = time.time()

        model.train()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if (batch_idx+1) % config["trainer"]["print_every"] == 0 and batch_idx > 1:
                print("  >> Iteration [%d]/%d, time consumed is [%.01fs] "%(batch_idx+1,  num_train_iter_per_epoch, time.time()-iter_time_start))
                iter_time_start = time.time()

            iter_info = train_iter(inputs=inputs,
                                    labels=labels,
                                    loss_func=loss_func,
                                    model=model,
                                    optimizer=optimizer,
                                    config=config,
                                    device=device)
            
            total_iter += 1
            
            for each_key in iter_info.keys():
                epoch_info[each_key] += iter_info[each_key]


        # ==== Print Training Stats
        epoch_info["train_loss"] = epoch_info["train_loss"] / num_train_iter_per_epoch
        print("  >> >> Training Loss [%.03f]" % (epoch_info["train_loss"]))

        summary["train_loss"].append(epoch_info["train_loss"])

        # ==== Reaching Checkpoint
        print("  >> Checkpoint Training time: [%.01fs]" % float(time.time()-time_start))
        time_start = time.time()

        # ==== Save checkpoint
        if not config["debug"]:
            save_model_ckp(config, model=model, 
                        epoch=epoch, iter_num=total_iter,
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        save_dir=save_root)
        
        # ==== Epoch Evaluation ==== 
        print("Epoch [%d] starts evaluation." % (epoch))

        model.eval()
        # ==== Evaluate on validation set
        val_result = validate(val_loader, model, loss_func, device)

        # ==== Print Validation Stats
        print("  >> >> Validation Metric [%.03f]" % (val_result["val_metric"]))

        summary["val_metric"].append(val_result["val_metric"])

        # ==== Plot Figures
        plot_train_val_eval(summary_dict=summary, save_dir=fig_dir, save_name="Train_Val_Loss_Plot", config=config)

        # ==== Save summary 
        file_name = os.path.join(save_root, "summary.npy")
        np.save(file_name, summary)

        # ==== Save checkpoint with least validation loss
        if summary["min_val_metric"] > val_result["val_metric"]:
            summary["min_val_metric"] = val_result["val_metric"]
            print("Minimal validation loss updated!")
            if not config["debug"]:
                save_model_ckp(config, model=model, 
                            epoch=epoch, iter_num=total_iter,
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            save_dir=mv_ckp_dir)

        # ==== Learning rate scheduler ==== 
        if config["optimizer"]["lr_scheduler"] != "RedusceLROnPlateau":
            scheduler.step()

        epoch += 1


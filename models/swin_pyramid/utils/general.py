import torch, os
from torch import nn
import random
import numpy as np

from models.swin_transformer import SwinTransformer

def seed_everything(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_device(config):
    if not config["debug"]:
        assert torch.cuda.is_available(), "CPU only is not an option for this project."
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
    return device


def build_model(config, 
                device=None):
    assert device is not None, "Specify your device please."
    model_cfg = config["model"]

    if model_cfg["type"] == "swin_transformer":
        model = SwinTransformer()

        if model_cfg["pretrained"]:
            ckp_content = torch.load(model_cfg["dir"])
            model_state_dict = ckp_content['model']
            model.load_my_state_dict(model_state_dict)
            print("Pretrained model loaded!")
            
        model = model.to(device)
    else:
        raise RuntimeError("The author is lazy and did not implement another model yet.")
    return model


def get_optimizer(config, model):
    opt = config["optimizer"]
    opt_type = opt["type"]

    if opt_type ==  "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=opt["lr"], 
                                     weight_decay=opt["weight_decay"])
    elif opt_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt["lr"], 
                                     weight_decay=opt["weight_decay"])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt["lr"],
                                    momentum=opt["momentum"],
                                    weight_decay=opt["weight_decay"])
    else:
        raise RuntimeError("The author did not implement other optimizers yet.")
    return optimizer 


def get_scheduler(config, optimizer):

    opt = config["optimizer"]

    if opt["lr_scheduler"] == "Linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                      start_factor=1, 
                                                      end_factor=0.01,
                                                      total_iters=100)
    # elif opt["lr_scheduler"] == "Cosine":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                            T_max=n_epoch)
    elif opt["lr_scheduler"] == "Exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.95)
    elif opt["lr_scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=40, 
                                                    gamma=0.1)
    elif opt["lr_scheduler"] == "RedusceLROnPlateau":
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    else:
        raise RuntimeError("The author did not implement other scheduler yet.")
    return scheduler

def get_loss_func(config):
    name = config["loss"]["func"]
    if name == "CE-smooth":
        loss_func = nn.CrossEntropyLoss(label_smoothing=config["loss"]["smoothing"])
    elif name == "CE":
        loss_func = nn.CrossEntropyLoss()
    elif name == "MSE":
        loss_func = nn.MSELoss()
    else:
        raise RuntimeError("Unimplemented Loss Type")
    return loss_func

def initialize_epoch_info(config):
    epoch_info = {
        "train_loss": 0
    }
    return epoch_info

def load_dict(root_dir, name):
    summary_file = os.path.join(root_dir, name)
    summary_data = np.load(summary_file, allow_pickle=True)
    saved_dict = summary_data.item()
    return saved_dict
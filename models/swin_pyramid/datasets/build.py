from datasets.open_monkey import OpenMonkeyDataset
import torch

from torchvision.transforms import Compose, Normalize

def get_transform(config, mode):
    transform = Compose([
        Normalize((128, 128, 128),
                  (255, 255, 255))
    ]) if mode[-1] != "v" else None
    return transform

def get_dataset(config, mode):
    data_cfg = config["dataset"]
    transform = get_transform(config=config, mode=mode)

    if mode[-1] == "v": mode = mode[:-1]
    dataset = OpenMonkeyDataset(root=data_cfg["dir"], mode=mode, transform=transform)

    if not data_cfg["subset"]:
        return dataset
    else:
        indices = list(range(0, len(dataset), data_cfg["ds_skip"]))
        return torch.utils.data.Subset(dataset, indices)

def get_dataloader(config, mode, shuffle=True):
    dataset = get_dataset(config=config, mode=mode)

    if mode[-1] == "v": mode = mode[:-1]
    if mode == "train":
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=config["trainer"]["train_batch_size"],
                                                  shuffle=shuffle)
    elif mode in ["test", "val"]:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=config["trainer"]["test_batch_size"],
                                                  shuffle=shuffle)
    else:
        raise RuntimeError("Dataset mode not implemented yet.")
    
    return data_loader




from datasets.open_monkey import OpenMonkeyDataset
import torch

def get_transform(config, mode):
    # == TO DO == 
    transform = None
    return transform

def get_dataset(config, mode):
    data_cfg = config["dataset"]
    transform = get_transform(config=config, mode=mode)
    if data_cfg["name"] == "open_monkey":
        dataset = OpenMonkeyDataset(root=data_cfg["dir"], mode=mode, transform=transform)
    else:
        raise RuntimeError("Dataset not implemented yet.")

    if not data_cfg["subset"]:
        return dataset
    else:
        indices = list(range(0, len(dataset), data_cfg["ds_skip"]))
        return torch.utils.data.Subset(dataset, indices)

def get_dataloader(config, mode):
    dataset = get_dataset(config=config, mode=mode)

    if mode == "train":
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=config["trainer"]["train_batch_size"],
                                                  shuffle=True)
    elif mode in ["test", "val"]:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=config["trainer"]["test_batch_size"],
                                                  shuffle=True)
    else:
        raise RuntimeError("Dataset mode not implemented yet.")
    
    return data_loader




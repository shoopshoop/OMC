import torch, os


def train_iter(inputs, labels, loss_func, model,
               optimizer, config, device):

    inputs = inputs.to(device)
    labels = labels.to(device)

    pred_heatmaps = model(inputs)
    loss = loss_func(labels, pred_heatmaps)
    loss.backward()

    if config["trainer"]["clip_grad"]:
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)

    optimizer.step()
    
    iter_info = {
        "train_loss": loss.item()
    }
    return iter_info



def save_model_ckp(config, model, epoch, iter_num,
                   optimizer, scheduler, save_dir):
    if config["data_parallel"]:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    checkpoint = {"epoch": epoch,
                  "iter": iter_num,
                  "model_state_dict": model_state_dict,
                  "optimizer": config["opts"]["optimizer"],
                  "scheduler": scheduler,
                  "optimizer_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))

@torch.no_grad()
def validate(dataloader, model):
    #  == to do ==
    result = {}
    return result
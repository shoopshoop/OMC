import json, os, argparse, time
import torch

from datasets.build import get_dataloader, get_dataset
from utils.general import build_model, set_device

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

    device = set_device(config)
    model = build_model(config, device)

    checkpoint_path = os.path.join(config["ckp_dir"], "checkpoint.pth")
    cpk_content = torch.load(checkpoint_path)
    model_state_dict = cpk_content["model_state_dict"]
    model.load_state_dict(model_state_dict)

    test_loader = get_dataloader(config, "test", shuffle=False)
    testv_data = get_dataset(config, "testv")

    model.eval()
    with torch.no_grad():
        num = len(test_loader)
        for iter, (inputs, idx) in enumerate(test_loader):
            if iter % 30 == 0: print("Iter %d (Total %d)" % (iter, num))
            inputs = inputs.to(device)
            pred_heatmap = model(inputs)
            for heatmap, each_idx in zip(pred_heatmap, idx):
                landmark = []
                for layer in heatmap[1:,:,:]:
                    index = (layer==torch.max(layer)).nonzero().cpu().tolist()[0]
                    landmark += index 
                testv_data.save_landmarks(idx_list=[each_idx.cpu().tolist()], landmarks=[landmark])
            break
    
    path = ""
    testv_data.write_landmarks_to_file(path)



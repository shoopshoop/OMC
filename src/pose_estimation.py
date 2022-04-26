import json, os, argparse, time






if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", required=True, type=str,
    #                     help="Path to the json config file.")
    parser.add_argument("--config", type=str,
                        default="./configs/config_local.json",
                        help="Path to the json config file.")
    parser.add_argument("--machine", type=str,
                        default="pc")
    args = parser.parse_args()
    cfg = json.load(open(args.config, "r"))
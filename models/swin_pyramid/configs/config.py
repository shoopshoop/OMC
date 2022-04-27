from datetime import date, datetime
import time, os, json
import numpy as np
import torch
import random
import sys

def update_config(config, args):
    config["data_parallel"] = True if torch.cuda.device_count() > 1 else False
    config["machine"] = args.machine

def save_exp_info(config):
    if config["resume_training"]:
        save_root = config["ckp_dir"]
    else:
        save_root = config["save_root"]
        save_root = create_root_dir(save_root)
        config["ckp_dir"] = save_root

    # Save console output
    filepath = os.path.join(save_root, "stdout.txt")
    sys.stdout = Logger(filepath)

    # print_config(config)

    # Save Exp Settings as Json File
    resume_training = config["resume_training"]
    config["resume_training"] = True # Always save config with resume_training being true

    exp_config_file = os.path.join(save_root, "Exp_Config.json")
    with open(exp_config_file, "w") as outfile:
        json.dump(config, outfile, indent=4)

    config["resume_training"] = resume_training
    return save_root, config


def save_dict_to_json(dict_to_save, dir, name):
    save_file_name = os.path.join(dir, name)
    with open(save_file_name, "w") as outfile:
        json.dump(dict_to_save, outfile, indent=4)

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    
    def __del__(self):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

def create_root_dir(root_name):
    today = date.today()
    date_stamp = today.strftime("%Y-%b-%d")
    print("  Experiment Date: ", date_stamp)
    save_root = os.path.join(root_name,
                             date_stamp)
    create_dir_success = False
    for _ in range(10):
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        save_dir = os.path.join(save_root,
                                current_time)
        if os.path.exists(save_dir):
            wait_n = np.random.randint(low=1, high=5)
            time.sleep(wait_n)
        else:
            try:
                os.makedirs(save_dir, exist_ok=False)
                create_dir_success = True
                break
            except:
                pass
    assert create_dir_success, "Create Log unsuccessful."
    print("  Experiment Time: ", current_time)
    return save_dir
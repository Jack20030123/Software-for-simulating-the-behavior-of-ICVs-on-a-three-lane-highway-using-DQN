import glob
import json
import logging
import os
from multiprocessing import Process, Queue
from matplotlib import pyplot as plt
import pandas as pd
def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
def plot(q):
    while True:
        try:
            data = q.get()
            title, xlabel, ylabel, x, y, save_path = data
            plt.figure(figsize=(40, 10), dpi=60)
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(e)
class PlotUtil:
    def __init__(self,):
        self.queue=Queue()
        self.init_process()
    def init_process(self):
        self.process=Process(target=plot,args=(self.queue,))
        self.process.start()
    def push_task(self,title,xlabel,ylabel,x,y,save_path):
        self.queue.put((title,xlabel,ylabel,x,y,save_path))
def print_args(logger,opt,config):
    logger.info(
        f"训练参数配置：\n"
        f"    save_dir:{opt.save_dir}\n"
        f"    train:{opt.train}\n"
        f"    test_model_dir:{opt.test_model_dir}\n"
        f"    episodes:{opt.episodes}\n"
        f"    num_agent:{opt.num_agent}\n"
        f"    hidden_size:{opt.hidden_size}\n"
        f"    batch_size:{opt.batch_size}\n"
        f"    lr:{opt.lr}\n"
        f"    gamma:{opt.gamma}\n"
        f"    memory_capacity:{opt.memory_capacity}\n"
        f"    copy_step:{opt.copy_step}\n"
        f"    action_space:{opt.action_space}\n"
        f"    desc:{opt.desc}\n"
        f"    render_mode:{opt.render_mode}\n"
        f"    input_size:{opt.input_size}\n"
        f"    output_size:{opt.output_size}\n"
        f"    exploration_decay:{opt.exploration_decay}\n"
        f"    train_memory_capacity:{opt.train_start_memory_capacity}\n"
        f"    save_step:{opt.save_step}\n"
        f"    min_exploration:{opt.min_exploration}\n"
        f"    max_exploration:{opt.max_exploration}\n")
    logger.info("游戏环境配置：\n  "+config.__str__())
def save_config(config_save_dir,opt,multi_highway_config):
    os.makedirs(config_save_dir,exist_ok=True)
    with open(os.path.join(config_save_dir,"multi_highway_config.json"),"w",encoding="utf-8") as f:
        json.dump(multi_highway_config,f)
    opt_dict=opt.__dict__
    with open(os.path.join(config_save_dir,"opt.json"),"w",encoding="utf-8") as f:
        json.dump(opt_dict,f)
def load_config(config_save_dir):
    with open(os.path.join(config_save_dir,"multi_highway_config.json"),"r",encoding="utf-8") as f:
        multi_highway_config=json.load(f, object_hook=int_keys_hook)
    with open(os.path.join(config_save_dir,"opt.json"),"r",encoding="utf-8") as f:
        opt_dict=json.load(f)
    return opt_dict,multi_highway_config
def int_keys_hook(obj):
    return {int(key) if key.isdigit() else key: value for key, value in obj.items()}
def get_log_path(log_root="run", sub_dir="exp"):
    os.makedirs(log_root, exist_ok=True)
    files = glob.glob(os.path.join(log_root,f"{sub_dir}")+"*", recursive=False)
    log_dir = os.path.join(log_root, f"{sub_dir}{len(files)+1}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
def logger_setting(logs_dir,file_handler_level=logging.DEBUG,stream_handler_level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    log_file = os.path.join(logs_dir, "run.log")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(file_handler_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_handler_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
if __name__ == '__main__':
    print(load_config("runs/train/exp1/models"))
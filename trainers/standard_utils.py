import os
import yaml
import argparse
import importlib
import csv
import time
import re
import sys
import os.path as osp
import numpy as np



class AverageMeter(object):
    """
    Computes and stores the average and current value for metric tracking.
    
    Args:
        name (str): Name of the metric
        fmt (str): Format string for printing (default: ':f')
    
    Methods:
        reset(): Reset all accumulated values to zero
        update(val, n=1): Update with new value(s), optionally with count n
        __str__(): Return formatted string representation
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def dict2namespace(config):
    """
    Recursively convert dictionary to argparse.Namespace object.
    
    Args:
        config (dict or argparse.Namespace): Configuration dictionary or existing Namespace
    
    Returns:
        argparse.Namespace: Converted namespace with dot-notation access
    """
    if isinstance(config, argparse.Namespace):
        return config
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace




def load_imf(log_path, config_fpath=None, ckpt_fpath=None, epoch=None, verbose=False, return_trainer=False, return_cfg=False):
    """
    Load implicit function (neural network) from checkpoint and config.
    
    Args:
        log_path (str): Path to experiment logs directory
        config_fpath (str): Path to config.yaml file (default: auto-locate)
        ckpt_fpath (str): Path to checkpoint file (default: latest.pt)
        epoch (int): Specific epoch checkpoint to load (default: latest)
        verbose (bool): Print debug information (default: False)
        return_trainer (bool): Return trainer object instead of just network (default: False)
        return_cfg (bool): Also return configuration (default: False)
    
    Returns:
        tuple: (imf, cfg) or (trainer, cfg) if return_trainer=True
    """
    # Load configuration
    if config_fpath is None:
        config_fpath = osp.join(log_path, "config", "config.yaml")
    with open(config_fpath) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.Loader))
    cfg.save_dir = "logs"

    # Load pretrained checkpoints
    ep2file = {}
    last_file, last_ep = osp.join(log_path, "latest.pt"), -1
    if ckpt_fpath is not None:
        last_file = ckpt_fpath
    else:
        ckpt_path = osp.join(log_path, "checkpoints")
        if osp.isdir(ckpt_path):
            for f in os.listdir(ckpt_path):
                if not f.endswith(".pt"):
                    continue
                ep = int(f.split("_")[1])
                if verbose:
                    print(ep, f)
                ep2file[ep] = osp.join(ckpt_path, f)
                if ep > last_ep:
                    last_ep = ep
                    last_file = osp.join(ckpt_path, f)
            if epoch is not None:
                last_file = ep2file[epoch]
    print(last_file)

    trainer_lib = importlib.import_module("trainers.HeatStep")
    trainer = trainer_lib.Trainer(cfg)
    trainer.resume(last_file)
    
    if return_trainer:
        return trainer, cfg
    else:
        imf = trainer.net
        del trainer
        return imf, cfg
    
def load_imf_mult(cfg, ckpt_path = None, load_both_nets = [False, False]):
    """
    Load dual networks (SDF and phase-field) from checkpoint with flexible fallback loading.
    
    Args:
        cfg (argparse.Namespace): Configuration with models.decoder and models.decoder_pf
        ckpt_path (str): Path to checkpoint directory or file (default: None)
        load_both_nets (list): [load_SDF, load_pf] flags for selective network loading
    
    Returns:
        tuple: (net_SDF, net_pf) both on CUDA device
    """
    lib = importlib.import_module(cfg.models.decoder.type)
    lib_pf= importlib.import_module(cfg.models.decoder_pf.type)
    
    # instantiate two separate networks: SDF and phase-field
    net_SDF_org = lib.Net(cfg, cfg.models.decoder)
    net_pf_org = lib_pf.Net(cfg, cfg.models.decoder_pf)
    net_SDF = lib.Net(cfg, cfg.models.decoder)
    net_pf = lib_pf.Net(cfg, cfg.models.decoder_pf)
    loaded_keys = []
    if(ckpt_path != None):
        try:
            ckpt_file = None
            if os.path.isdir(ckpt_path):
                latest = os.path.join(ckpt_path, "latest.pt")
                if os.path.isfile(latest):
                    ckpt_file = latest
                else:
                    cands = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith(".pt")]
                    if cands:
                        ckpt_file = max(cands, key=os.path.getmtime)
            elif os.path.isfile(ckpt_path):
                ckpt_file = ckpt_path

            if ckpt_file is not None:
                ckpt = torch.load(ckpt_file, map_location="cpu")

                # Prefer explicit keys for both nets
                if 'net_SDF' in ckpt:
                    try:
                        net_SDF.load_state_dict(ckpt['net_SDF'], strict=False)
                        loaded_keys.append('net_SDF')
                    except Exception:
                        pass
                if 'net_pf' in ckpt:
                    try:
                        net_pf.load_state_dict(ckpt['net_pf'], strict=False)
                        loaded_keys.append('net_pf')
                    except Exception:
                        pass

                # Backwards-compatible fallbacks (only try if not already loaded)
                if not loaded_keys:
                    candidates = ['net', 'state_dict', 'model']
                    for key in candidates:
                        if key in ckpt and isinstance(ckpt[key], dict):
                            try:
                                # try loading into SDF first
                                net_SDF.load_state_dict(ckpt[key], strict=False)
                                loaded_keys.append(key + "->net_SDF")
                                break
                            except Exception:
                                # try loading into pf
                                try:
                                    net_pf.load_state_dict(ckpt[key], strict=False)
                                    loaded_keys.append(key + "->net_pf")
                                    break
                                except Exception:
                                    pass

                # If checkpoint itself looks like a plain state_dict, try direct loading
                if not loaded_keys and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    try:
                        net_SDF.load_state_dict(ckpt, strict=False)
                        loaded_keys.append('direct-state-dict->net_SDF')
                    except Exception:
                        try:
                            net_pf.load_state_dict(ckpt, strict=False)
                            loaded_keys.append('direct-state-dict->net_pf')
                        except Exception:
                            pass

                # Report results
                if len(loaded_keys) == 2:
                    print(f"Loaded 2 networks from checkpoint: {ckpt_file} (keys: {loaded_keys})")
                elif len(loaded_keys) == 1:
                    print(f"Loaded 1 network from checkpoint: {ckpt_file} (key: {loaded_keys[0]})")
                else:
                    print(f"No networks loaded from checkpoint: {ckpt_file} (available keys: {list(ckpt.keys())})")
            else:
                print(f"No checkpoint found at {ckpt_path}, skipping load.")
        except Exception as e:
            print("Warning: failed to load checkpoint:", e)
        if load_both_nets[0] and load_both_nets[1]:
            return net_SDF.cuda(), net_pf.cuda()
        elif load_both_nets[0]:
            return net_SDF.cuda(), net_pf_org.cuda()
        elif load_both_nets[1]:
            return net_SDF_org.cuda(), net_pf.cuda()
        else:
            return net_SDF_org.cuda(), net_pf_org.cuda()
    else:
        return net_SDF.cuda(), net_pf.cuda()

def parse_hparams(hparam_lst):
    """
    Parse hyperparameter list from command-line format (key=value pairs).
    
    Args:
        hparam_lst (list): List of strings like ['lr=0.001', 'batch_size=32']
    
    Returns:
        tuple: (dict of parsed params, formatted string for logging)
    """
    print("=" * 80)
    print("Parsing:", hparam_lst)
    out_str = ""
    out = {}
    for i, hparam in enumerate(hparam_lst):
        hparam = hparam.strip()
        k, v = hparam.split("=")[:2]
        k = k.strip()
        v = v.strip()
        print(k, v)
        out[k] = v
        out_str += "%s=%s_" % (k, v.replace("/", "-"))
    print(out)
    print(out_str)
    print("=" * 80)
    return out, out_str



def update_cfg_with_hparam(cfg, k, v):
    """
    Update configuration namespace with single hyperparameter using dot-notation path.
    
    Args:
        cfg (argparse.Namespace): Configuration object to modify
        k (str): Dot-notation key path (e.g., 'models.decoder.dim')
        v (str): Value to set (will be converted to original type)
    
    Returns:
        None: Modifies cfg in-place
    """
    k_path = k.split(".")
    cfg_curr = cfg
    for k_curr in k_path[:-1]:
        assert hasattr(cfg_curr, k_curr), "%s not in %s" % (k_curr, cfg_curr)
        cfg_curr = getattr(cfg_curr, k_curr)
    k_final = k_path[-1]
    assert hasattr(cfg_curr, k_final), \
        "Final: %s not in %s" % (k_final, cfg_curr)
    v_type = type(getattr(cfg_curr, k_final))
    setattr(cfg_curr, k_final, v_type(v))



def update_cfg_hparam_lst(cfg, hparam_lst):
    """
    Update configuration with multiple hyperparameters from list.
    
    Args:
        cfg (argparse.Namespace): Configuration object to modify
        hparam_lst (list): List of 'key=value' strings
    
    Returns:
        tuple: (updated_cfg, formatted_string for logging)
    """
    hparam_dict, hparam_str = parse_hparams(hparam_lst)
    for k, v in hparam_dict.items():
        update_cfg_with_hparam(cfg, k, v)
    return cfg, hparam_str



def dict2namespace(config):
    """
    Recursively convert dictionary to argparse.Namespace object.
    
    Args:
        config (dict or argparse.Namespace): Configuration dictionary or existing Namespace
    
    Returns:
        argparse.Namespace: Converted namespace with dot-notation access
    """
    if isinstance(config, argparse.Namespace):
        return config
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

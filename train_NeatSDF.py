import os
import yaml
import time
import math
import torch
import importlib
import numpy as np
import os.path as osp
from pathlib import Path
import argparse
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from trainers.standard_utils import AverageMeter, dict2namespace
from trainers.helper import comp_weights, create_initial_grid


class PointBatchLoader:
    """
    Batched point cloud loader that ensures every point is seen exactly M times
    per epoch across a fixed number of steps with a fixed batch size.

    Usage:
        loader = PointBatchLoader(pts, weights, batch_size=1000, steps_per_epoch=500)
        loader.start_epoch()           # shuffle at epoch start
        for step in range(500):
            batch_pts, batch_w = loader.get_batch(step)
    """

    def __init__(self, pts, weights, batch_size=1000, steps_per_epoch=500, device='cuda'):
        """
        Args:
            pts: (N, dim) point cloud tensor on device
            weights: (N,) or (N,1) per-point weights on device
            batch_size: number of points per step
            steps_per_epoch: number of steps in one epoch
            device: torch device
        """
        self.pts = pts
        self.weights = weights.view(-1) if weights.dim() > 1 else weights
        self.N = pts.shape[0]
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.device = device

        self.total_samples = batch_size * steps_per_epoch  # e.g. 500,000
        self.M = self.total_samples // self.N              # repeats per point
        if self.M < 1:
            self.M = 1

        # How many index slots we fill with full repeats
        self._full = self.M * self.N
        # How many extra slots to pad to reach total_samples exactly
        self._pad = self.total_samples - self._full

        print(f"[PointBatchLoader] N={self.N}, batch_size={batch_size}, "
              f"steps={steps_per_epoch}, M={self.M}, "
              f"pad={self._pad} ({100*self._pad/self.total_samples:.1f}%)")

        self._indices = None

    def start_epoch(self):
        """Call once at the beginning of each epoch. Builds a shuffled index tensor."""
        # M full copies of [0..N-1]
        base = torch.arange(self.N, device=self.device).repeat(self.M)

        if self._pad > 0:
            # pad with random indices (these points get seen M+1 times — unavoidable remainder)
            extra = torch.randint(0, self.N, (self._pad,), device=self.device)
            base = torch.cat([base, extra], dim=0)

        # global shuffle
        perm = torch.randperm(base.shape[0], device=self.device)
        self._indices = base[perm]

    def get_batch(self, step):
        """
        Get the batch for a given step index.

        Returns:
            batch_pts: (batch_size, dim)
            batch_weights: (batch_size,)
        """
        start = step * self.batch_size
        idx = self._indices[start:start + self.batch_size]

        return (
            torch.index_select(self.pts, 0, idx),
            torch.index_select(self.weights, 0, idx)
        )

def get_args():
    """
    Load configuration file based on point cloud dimension.
    
    Args:
        None (reads from command-line --dim argument)
    
    Returns:
        argparse.Namespace: Configuration object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=3, help="Point cloud dimension (2 or 3)")
    args = parser.parse_args()
    
    # Select config based on dimension
    if args.dim == 2:
        config_file = "configs/NeuralSDFsfromPF2D.yaml"
    else:
        config_file = "configs/NeuralSDFsfromPF3D.yaml"
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    
    os.makedirs(osp.join(config.save_dir, 'config'), exist_ok=True)
    with open(osp.join(config.save_dir, "config", "config.yaml"), "w") as outf:
        yaml.dump(config, outf)

    return config
        
def load_point_data(base_path):
    """
    Load point data from either .txt or .csv file.
    Tries .csv first, then .txt if .csv doesn't exist.
    
    Args:
        base_path: path without extension (e.g., "/home/weidemaier/NeatSDF/data/points")
    
    Returns:
        numpy array of shape (N, 3) or (N, 6) if normals are included
    """
    # Try .csv first
    csv_path = base_path + ".csv"
    if os.path.isfile(csv_path):
        print(f"[✓] Loading points from CSV: {csv_path}")
        try:
            pts_data = np.float32(np.loadtxt(csv_path, skiprows=1, delimiter=","))
        except Exception as e:
            try: 
                pts_data = np.float32(np.loadtxt(csv_path, skiprows=0, delimiter=" "))
            except Exception as e:
                print(f"[✗] Error loading CSV file: {e}")
                raise
        return pts_data
            
    # Try .txt as fallback
    txt_path = base_path + ".txt"
    if os.path.isfile(txt_path):
        print(f"[✓] Loading points from TXT: {txt_path}")
        try: 
            pts_data = np.float32(np.loadtxt(txt_path, skiprows=1, delimiter=","))
        except Exception as e:
            try:
                pts_data = np.float32(np.loadtxt(txt_path, skiprows=0, delimiter=" "))
            except Exception as e:
                print(f"[✗] Error loading TXT file: {e}")
                raise
        return pts_data
    
    # Neither file found
    raise FileNotFoundError(f"Could not find point data at {csv_path} or {txt_path}")


def main_worker(cfg):
    # basic setup
    cudnn.benchmark = True
    ##############################################################################
    dim = cfg.models.decoder.dim
    
    base_path = str(
        Path(__file__).resolve().parent
    ) + "/"

    pts_data =  load_point_data(base_path + cfg.input.point_path)
    
    # Check if normals are provided (N, 6) format
    pts = torch.tensor(pts_data[:, :3] if pts_data.ndim == 2 else pts_data[:3]).to('cuda')
    weights = comp_weights(pts.detach().cpu().numpy() ,0.001, dim)
    weights = torch.tensor(np.float32(weights)).cuda()
    ptcld_size = pts.shape[0]
    weights = (ptcld_size/5000)*weights
    pt_loader = PointBatchLoader(
        pts, weights,
        batch_size=5000,
        steps_per_epoch=500,
        device='cuda'
    )
    
    if(dim == 2):
        bounds = tuple(cfg.trainer.training_params.grid_max_2d)
    else:
        bounds = tuple(cfg.trainer.training_params.grid_max_3d)
    
    grid_resolution = cfg.sampling.init_grid_res
    midpoints, widths = create_initial_grid(bounds, grid_resolution, device='cuda', dim=dim)
    flags = torch.zeros(grid_resolution**dim, 1).cuda()  # level 0 = no refinement
    
    midpoints0 = midpoints
    widths0 = widths
    flags0 = flags
    
    GRID_MAX = bounds
    h = GRID_MAX[0] / cfg.sampling.init_grid_res
    
    ##############################################################################
    writer = SummaryWriter(log_dir=cfg.save_dir)
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg)

    start_epoch = 0
    start_time = time.time()

    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs + start_epoch))
    step = 0
    duration_meter = AverageMeter("Duration")
    loader_meter = AverageMeter("Loader time")
    best_val = np.Infinity

    ### start actual training loop
    trainer.save_run_state(cfg)
    
    trainer.precompute_sampling_constants(
            midpoints,
            flags,
            cfg.input.parameters.bs,
            h,
            dim=dim,
            device='cuda'
        )
    trainer.net_SDF.train()
    for p in trainer.net_SDF.parameters():
        p.requires_grad_(True)
    
    trainer.net_pf.eval()
    for p in trainer.net_pf.parameters():
        p.requires_grad_(False)

    for epoch in range(start_epoch, cfg.trainer.epochs + start_epoch):
        # train for one epoch
        loader_start = time.time()

        # Shuffle point cloud indices for this epoch
        pt_loader.start_epoch()
        ignore_pf = 5
        if epoch == ignore_pf:
            print(f"Epoch {epoch}: Starting to train PF component.")
            trainer.train_sdf, trainer.train_pf = True, True
            trainer.net_pf.train()
            for p in trainer.net_pf.parameters():
                p.requires_grad_(True)
        elif epoch == 20:
            print(f"Epoch {epoch}: Stopping PF training, finetuning SDF.")
            trainer.train_sdf, trainer.train_pf = True, False
            trainer.net_pf.eval()
            for p in trainer.net_pf.parameters():
                p.requires_grad_(False)

        for batchnumber in range(500):

            step = batchnumber + 500 * epoch + 1

            # Get this step's point batch
            batch_pts, batch_w = pt_loader.get_batch(batchnumber)

            ### evaluate loss
            logs_info = trainer.update(cfg, epoch, step, batch_pts, batch_w)
            
            if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
                duration = time.time() - start_time
                duration_meter.update(duration)
                start_time = time.time()
                print("Epoch %d Batch [%2d/%2d] Time [%3.2fs]"
                      " Loss %2.5f"
                      % (epoch, batchnumber, 500, duration_meter.avg, logs_info['loss']))
                trainer.log_train(
                    logs_info,
                    writer=writer, epoch=epoch, step=step)

            # Reset loader time
            loader_start = time.time()
        val = trainer.validate(epoch, cfg, midpoints0, flags0)
        
        midpoints = val['ref_mid']
        flags = val["levels"]
        val_loss = val['loss']

        if dim == 2:
            trainer.sch_pf.step(val_loss)
            trainer.sch_SDF.step(val_loss)
        elif dim == 3:
            if trainer.train_pf:
                trainer.sch_pf.step()
            trainer.sch_SDF.step()
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step, vis = True)
        
    writer.close()


cfg = get_args()

main_worker(cfg)

import os
import torch
import os.path as osp

from trainers.diff_ops import gradient
from trainers.base_trainer import BaseTrainer
from trainers.utils import set_random_seed
from trainers.standard_utils import load_imf_mult
from trainers.helper import adaptive_refinement 
from trainers.pf_loss_functionals import *
from trainers.vis_utils import vis_sdf_and_pf



class Trainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # load pretrained SDF if desired; if path is empty, will initialize randomly
        ckpt_path = "configs/2D_init/" 

        self.net_SDF, self.net_pf = load_imf_mult(cfg, ckpt_path, [True, False])

        print("Net_SDF:")
        print(self.net_SDF)
        print("Net_pf:")
        print(self.net_pf)

        # The optimizers
        cfgopt_sdf = getattr(self.cfg.trainer, "opt_sdf", getattr(self.cfg.trainer, "opt", None))
        cfgopt_pf = getattr(self.cfg.trainer, "opt_pf", getattr(self.cfg.trainer, "opt", None))

        if cfgopt_sdf is None or cfgopt_pf is None:
            raise ValueError("Optimizer configuration missing in cfg.trainer")

        lr_sdf = float(cfgopt_sdf.lr)
        lr_pf = float(cfgopt_pf.lr)
        
        self.opt_SDF = torch.optim.Adam(self.net_SDF.parameters(), lr=lr_sdf,
                               betas=(cfgopt_sdf.beta1, cfgopt_sdf.beta2),
                               weight_decay=float(cfgopt_sdf.weight_decay))
        self.opt_pf = torch.optim.Adam(self.net_pf.parameters(), lr=lr_pf,
                               betas=(cfgopt_pf.beta1, cfgopt_pf.beta2),
                               weight_decay=float(cfgopt_pf.weight_decay))
        
        # schedulers for both optimizers
        self.sch_SDF = torch.optim.lr_scheduler.StepLR(
                self.opt_SDF,
                step_size=5,
                gamma=0.5,
            )
        self.sch_pf = torch.optim.lr_scheduler.StepLR(
                self.opt_pf,
                step_size=10,
                gamma=0.5,
            )
        
        # Prepare save directory
        os.makedirs(osp.join(cfg.save_dir, "checkpoints"), exist_ok=True)

        # add internal variables
        self._precomputed_scales = None
        self._precomputed_weights = None
        self._precomputed_cell_idx = None
        self._precomputed_levels = None
        self.train_sdf = True
        self.train_pf = False
        self._sdf_params = list(self.net_SDF.parameters())
        self._pf_params = list(self.net_pf.parameters())
    
        # samples to fix sign outside
        grid_width = cfg.trainer.training_params.grid_max_2d
        gw = torch.as_tensor(grid_width, dtype=torch.float32, device="cuda")
        self.x_outer = 2 * gw * torch.rand((1000, 2), device="cuda") - gw


    def precompute_sampling_constants(self, midpoints, flags, batch_size, h, dim=2, device='cuda'):
        """
        Precompute constant values for sampling that don't change during an epoch.
        Call this once per epoch after refinement.
        
        Parameters:
            midpoints (torch.Tensor): Box centers of shape (B, dim)
            flags (torch.Tensor): Refinement levels of shape (B,)
            batch_size (int): Batch size
            h (float): Base cell width
            dim (int): Spatial dimension
            device (str): Device
        """
        B = midpoints.shape[0]
        points_per_cell = (batch_size + B - 1) // B
        
        # Precompute cell indices (constant)
        self._precomputed_cell_idx = torch.arange(B, device=device).repeat_interleave(points_per_cell)
        self._precomputed_levels = flags[self._precomputed_cell_idx].to(torch.float32)
        
        # Precompute scales and weights (constant)
        base = 2.0 ** dim
        power_scale = torch.pow(2.0, self._precomputed_levels - 1.0)
        power_weights = torch.pow(base, -self._precomputed_levels)
        
        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=device)
        self._precomputed_scales = h_tensor / power_scale
        self._precomputed_weights = base * power_weights
        
        # Also store selected midpoints (constant)
        self._precomputed_midpoints = midpoints[self._precomputed_cell_idx]
    


    def sample_and_reweight_fast_cached(self, dim=2, fast=False, device='cuda'):
        """
        Fast sampling using precomputed constants. Only generates random offsets each call.
        
        Parameters:
            dim (int): Spatial dimension
            fast (bool): If True, reuse same offsets for all points in each cell
            device (str): Device
        
        Returns:
            tuple: (sampled_points, integration_weights)
        """
        total_points = self._precomputed_scales.shape[0]
        points_per_cell = total_points // (self._precomputed_cell_idx.max().item() + 1)
        
        # Only generate random offsets (fast operation)
        if fast:
            rand = torch.rand((points_per_cell, dim), device=device) - 0.5
            rand = rand.repeat(total_points // points_per_cell, 1)
        else:
            rand = torch.rand((total_points, dim), device=device) - 0.5
        
        # Apply precomputed scales to random offsets
        offsets = rand * self._precomputed_scales.view(-1, 1)
        
        # Return using precomputed midpoints and weights
        return self._precomputed_midpoints + offsets, self._precomputed_weights
    



    ### compute loss and update optimizers
    def update(self, cfg, epoch, step, pts, weights):
        device = "cuda"

        GRID_MAX = tuple(cfg.trainer.training_params.grid_max_2d)
        MASS_FACTOR = 100.0
        BENDING_FAKTOR = 15.0
        POINTCLOUD_FACTOR = 1.e+6
        eps = cfg.input.epsilon
        AMBROSIO_TORTORELLI_FACTOR = 1/5
        EIKONAL_FACTOR = 0.1
        BD_FACTOR = 2000.0

        if epoch <= 10:
            MASS_FACTOR = 100.0
            POINTCLOUD_FACTOR = 1.e+7
            BENDING_FAKTOR = 10.0
        elif epoch < 20:
            MASS_FACTOR = 175.0
        elif epoch < 30:
            MASS_FACTOR = 100.0 - 10 * (epoch - 20)
        elif epoch < 40:
            MASS_FACTOR = 10.0 - 1 * (epoch - 30)
        else:
            MASS_FACTOR = 1.0

        self.opt_SDF.zero_grad()
        self.opt_pf.zero_grad()

        x, w = self.sample_and_reweight_fast_cached(dim=2, fast=True, device=device)
        bs = x.shape[0]
        x.requires_grad_(True)

        neural_sdf = self.net_SDF(x)
        neural_pf = self.net_pf(x)

        sdf = neural_sdf.view(-1, 1)
        sdf_outer = self.net_SDF(self.x_outer).view(-1, 1)
        phasefield = neural_pf.view(-1, 1)
        grad_phasefield = gradient(phasefield, x)

        sdf_at_pointcloud = self.net_SDF(pts).view(-1, 1)

        grad_sdf = gradient(sdf, x)
        grad_sdf_squared = (grad_sdf * grad_sdf).sum(dim=-1)

        if epoch <= 1:
            pts = pts.requires_grad_(True)
            grad_sdf_surf = gradient(self.net_SDF(pts), pts)
            grad_sdf_surf_squared = (grad_sdf_surf * grad_sdf_surf).sum(dim=-1)
            hess_times_grad_sdf_surf = 0.5 * gradient(grad_sdf_surf_squared, pts)
            hess_times_grad_sdf = 0.5 * gradient(grad_sdf_squared, x)
        else:
            hess_times_grad_sdf = 0.5 * gradient(grad_sdf_squared, x)

        loss = 0.

        if epoch < 1:
            bending_loss_surf = BENDING_FAKTOR * bendingEnergySDF(
                torch.ones_like(sdf_at_pointcloud), None, hess_times_grad_sdf_surf, eps=eps, rect=GRID_MAX
            ).view(-1, 1)
            bending_loss = BENDING_FAKTOR * bendingEnergySDF(
                phasefield, None, hess_times_grad_sdf, eps=eps, rect=GRID_MAX
            ).view(-1, 1)
            loss += 0.55 * (w * bending_loss).mean() + 0.75 * bending_loss_surf.mean()
        else:
            bending_loss = BENDING_FAKTOR * bendingEnergySDF(
                phasefield, None, hess_times_grad_sdf, eps=eps, rect=GRID_MAX
            ).view(-1, 1)
            loss += (w * bending_loss).mean()

        eikonal_loss = EIKONAL_FACTOR * eikonalEnergy(grad_sdf, eps=eps, rect=GRID_MAX).view(-1, 1)
        loss += (w * eikonal_loss).mean()

        ambrosioTortorelli_loss = AMBROSIO_TORTORELLI_FACTOR * ambrosioTortorelli(
            phasefield, grad_phasefield, eps=eps, rect=GRID_MAX
        ).view(-1, 1)
        loss += (w * ambrosioTortorelli_loss).mean()

        pointcloud_loss = POINTCLOUD_FACTOR * pointcloudEnergy(sdf_at_pointcloud, weight=weights)
        loss += pointcloud_loss

        mass_loss = (eps / 0.001) * MASS_FACTOR * (torch.exp(-100 * torch.abs(sdf) ** 2).view(-1, 1)
                                                   + 3 * torch.exp(-100 * torch.abs(sdf)).view(-1, 1)
                                                   + torch.exp(-10 * torch.abs(sdf) ** 3).view(-1, 1))
        mass_loss += (eps / 0.001) * MASS_FACTOR * torch.exp(-1 * torch.abs(sdf) ** 2).view(-1, 1)
        loss += (w * mass_loss).mean()

        boundary_loss = BD_FACTOR * bd_loss_alternative(sdf_outer)
        loss += boundary_loss

        loss = 1 / 100 * loss
        
        loss.backward()

        if self.train_sdf:
            torch.nn.utils.clip_grad_norm_(self._sdf_params, max_norm=100.0)
            self.opt_SDF.step()
            self.opt_SDF.zero_grad(set_to_none=True)


        if self.train_pf:
            torch.nn.utils.clip_grad_norm_(self._pf_params, max_norm=1.)
            self.opt_pf.step()
            self.opt_pf.zero_grad(set_to_none=True)

        if step % 500 == 0:
            vis_sdf_and_pf(self.net_SDF, self.net_pf, bounds=(-1.2, 1.2),
                           save_path=osp.join(self.cfg.save_dir, f"vis_step_{step}.png"))

        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/bending_loss': bending_loss.mean().detach().cpu().item(),
            'scalar/eikonal_loss': (w * eikonal_loss).mean().detach().cpu().item(),
            'scalar/ambrosioTortorelli_loss': (w * ambrosioTortorelli_loss).mean().detach().cpu().item(),
            'scalar/pointcloud_loss': pointcloud_loss.mean().detach().cpu().item(),
            'scalar/mass_loss': (w * mass_loss).mean().detach().cpu().item(),
        }


    def log_train(self, train_info, writer=None,
                  step=None, epoch=None):
        if writer is None:
            return

        # Log training information to tensorboard
        writer_step = step if step is not None else epoch
        assert writer_step is not None
        for k, v in train_info.items():
            t, kn = k.split("/")[0], "/".join(k.split("/")[1:])
            if t not in ['scalar']:
                continue
            if t == 'scalar':
                writer.add_scalar('train/' + kn, v, writer_step)
        writer.add_scalar('train/learning_rate_SDF', self.opt_SDF.param_groups[0]["lr"], writer_step)
        writer.add_scalar('train/learning_rate_pf', self.opt_pf.param_groups[0]["lr"], writer_step)



    def validate(self, epoch, cfg, midpoints, lvl):
        device = "cuda"

        def SDF_abs(x):
            return torch.abs(self.net_SDF(x))
        refined_midpoints, refined_levels = adaptive_refinement(
            box_midpoints=midpoints, 
            refinement_lvl=lvl, 
            functions=[SDF_abs, self.net_pf], 
            threshold=[0.05, 0.25], 
            device=device, 
            dim=2
        )
        
        return {
            'loss': 0.0,
            'ref_mid': refined_midpoints,
            'levels': refined_levels} 



    def save(self, epoch=None, step=None, appendix=None, vis=False):
        d = {
            'opt_SDF': self.opt_SDF.state_dict(),
            'opt_pf': self.opt_pf.state_dict(),
            'net_SDF': self.net_SDF.state_dict(),
            'net_pf': self.net_pf.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        torch.save(d, osp.join(self.cfg.save_dir, "checkpoints", save_name))
        torch.save(d, osp.join(self.cfg.save_dir, "latest.pt"))


    
    def save_run_state(self, cfg):
        import os
        import shutil
        import inspect
        from datetime import datetime

        run_dir = os.path.join(cfg.save_dir, "run_state")
        os.makedirs(run_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # get the top-level executing script
        main_file = inspect.getmodule(inspect.stack()[-1][0]).__file__
        main_file2 = inspect.getfile(self.__class__)
        files = [main_file, main_file2]
        for i, f in enumerate(files):
            shutil.copy(
                f,
                os.path.join(run_dir, f"source_{i}_{ts}.py")
            )

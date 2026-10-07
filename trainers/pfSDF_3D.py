import os
import torch
import os.path as osp
from trainers.vis_utils import imf2mesh
from trainers.base_trainer import BaseTrainer
from trainers.utils import set_random_seed
from trainers.standard_utils import load_imf_mult
from trainers.helper import adaptive_refinement_3D
from trainers.pf_loss_functionals import *



class Trainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))
        # load pretrained SDF if desired; if path is empty, will initialize randomly
        ckpt_path = " "
        self.net_SDF, self.net_pf = load_imf_mult(cfg, ckpt_path, [True, False])
        self.net_pf = torch.compile(self.net_pf)
        print("Net_SDF:")
        print(self.net_SDF)
        print("Net_pf:")
        print(self.net_pf)

        # initialize optimizers
        cfgopt_sdf = getattr(self.cfg.trainer, "opt_sdf", getattr(self.cfg.trainer, "opt", None))
        cfgopt_pf = getattr(self.cfg.trainer, "opt_pf", getattr(self.cfg.trainer, "opt", None))

        if cfgopt_sdf is None or cfgopt_pf is None:
            raise ValueError("Optimizer configuration missing in cfg.trainer")

        lr_sdf = float(cfgopt_sdf.lr)
        lr_pf = float(cfgopt_pf.lr)
        
        self.opt_SDF = torch.optim.Adam(self.net_SDF.parameters(), lr=lr_sdf,
                               betas=(cfgopt_sdf.beta1, cfgopt_sdf.beta2),
                               weight_decay=float(cfgopt_sdf.weight_decay), eps = 1e-6)
        self.opt_pf = torch.optim.Adam(self.net_pf.parameters(), lr=lr_pf,
                               betas=(cfgopt_pf.beta1, cfgopt_pf.beta2),
                               weight_decay=float(cfgopt_pf.weight_decay))
        
        self.sch_SDF = torch.optim.lr_scheduler.StepLR(
                self.opt_SDF,
                step_size=20,
                gamma=0.1,
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
        self.batchsize = 0



    def precompute_sampling_constants(self, midpoints, flags, batch_size, h, dim=3, device='cuda'):
        B = midpoints.shape[0]
        points_per_cell = (batch_size + B - 1) // B
        total_points = points_per_cell * B
        self.batchsize = total_points  # Store for later use in sampling method
        self._precomputed_cell_idx = torch.arange(B, device=device).repeat_interleave(points_per_cell)
        self._precomputed_levels = flags[self._precomputed_cell_idx].to(torch.float32)
        
        power_scale = torch.pow(2.0, self._precomputed_levels)
        
        h_tensor = torch.as_tensor(h, dtype=torch.float32, device=device)
        self._precomputed_scales = h_tensor / power_scale

        # Cell volume = side_length^dim
        cell_volume = self._precomputed_scales ** dim
        # Normalize so sum(weights) == total_points
        self._precomputed_weights = (cell_volume / cell_volume.sum() * float(total_points)).view(-1)

        self._precomputed_midpoints = midpoints[self._precomputed_cell_idx]

    
    
    def sample_and_reweight_fast_cached(self, dim=3, fast=False, device='cuda'):
        """
        Fast sampling using precomputed constants. Only generates random offsets each call.
        """
        total_points = self._precomputed_scales.shape[0]
        points_per_cell = total_points // (self._precomputed_cell_idx.max().item() + 1)
        
        group = total_points // points_per_cell

        rand = torch.rand((points_per_cell, dim), device=device) - 0.5
        rand = rand.repeat_interleave(group, dim=0)
        offsets = rand * self._precomputed_scales
        
        return self._precomputed_midpoints + offsets, self._precomputed_weights
    


    ### compute loss and update optimizers
    def update(self, cfg, epoch, step, pts, weights):
        assert torch.cuda.is_bf16_supported()
        device = "cuda"
        improv_hess = epoch >= 20

        GRID_MAX = tuple(cfg.trainer.training_params.grid_max_3d)
        eps = cfg.input.epsilon

        MASS_FACTOR = 250.
        BENDING_FAKTOR = 1.
        POINTCLOUD_FACTOR = 1.e+6
        AMBROSIO_TORTORELLI_FACTOR = 1/20
        EIKONAL_FACTOR = 0.05  

        if epoch <= 10:
            MASS_FACTOR = 250
            AMBROSIO_TORTORELLI_FACTOR = 1/50
        elif epoch < 20:
            MASS_FACTOR = 250
            EIKONAL_FACTOR = 0.05 + 0.015 * (epoch - 10)
            POINTCLOUD_FACTOR = 1e6 + 4.5e6 * (epoch - 10)
            BENDING_FAKTOR = 1. + 0.15 * (epoch - 10)
            AMBROSIO_TORTORELLI_FACTOR = 1/30 + 5*(1/30) * (epoch - 10) / 10
        else:
            MASS_FACTOR = max(100, 250 - 20 * (epoch - 20))
            EIKONAL_FACTOR = 0.20
            POINTCLOUD_FACTOR = 5.e7
            BENDING_FAKTOR = 2.5
            AMBROSIO_TORTORELLI_FACTOR = 1/5

        MASS_FACTOR *= 2.

        # --- sampling ---
        x, w = self.sample_and_reweight_fast_cached(dim=3, fast=True, device=device)
        x.requires_grad_(True)
        w = w.view(-1, 1).detach()
        weights = weights.detach()

        # --- forward ---
        with torch.cuda.amp.autocast(dtype=torch.float32):
            sdf = self.net_SDF(x)
            pts = pts.requires_grad_(True)
            sdf_pts = self.net_SDF(pts)

            # --- first gradient (needs graph) ---
            grad_sdf = torch.autograd.grad(
                sdf, x,
                grad_outputs=torch.ones_like(sdf),
                create_graph=True,
                retain_graph=True
            )[0]

            grad_sdf_norm = torch.norm(grad_sdf, dim=-1)
            grad_sdf_sq = grad_sdf_norm * grad_sdf_norm

            # --- second gradient  ---
            hess_times_grad_sdf = 0.5 * torch.autograd.grad(
                grad_sdf_sq, x,
                grad_outputs=torch.ones_like(grad_sdf_sq),
                create_graph=True,   # MUST be True
                retain_graph=True
            )[0]

            # --- phase field ---
            if epoch >= 5:
                phasefield = self.net_pf(x).unsqueeze(-1)

                grad_phasefield = torch.autograd.grad(
                    phasefield, x,
                    grad_outputs=torch.ones_like(phasefield),
                    create_graph=False,
                    retain_graph=True
                )[0]
            else:
                with torch.no_grad():
                    phasefield = torch.full((self.batchsize, 1), 0.5, device=device)
                    grad_phasefield = torch.zeros((self.batchsize, 1), device=device)

            loss = torch.zeros((), device=device)

        # --- bending ---
        if improv_hess:
            bending_loss_1 = BENDING_FAKTOR * bendingEnergySDF(
                phasefield, None, hess_times_grad_sdf, eps=eps, rect=GRID_MAX
            ).view(-1, 1)

            phase_field_pts = self.net_pf(pts).unsqueeze(-1)

            # --- surface gradients ---
            grad_pts = torch.autograd.grad(
                sdf_pts, pts,
                grad_outputs=torch.ones_like(sdf_pts),
                create_graph=True,
                retain_graph=True
            )[0]

            grad_pts_norm = torch.norm(grad_pts, dim=-1)
            grad_pts_sq = grad_pts_norm * grad_pts_norm

            hess_times_grad_sdf_pts = 0.5 * torch.autograd.grad(
                grad_pts_sq, pts,
                grad_outputs=torch.ones_like(grad_pts_sq),
                create_graph=True,
                retain_graph=True
            )[0]

            bending_loss_2 = BENDING_FAKTOR * bendingEnergySDF(
                torch.ones_like(phase_field_pts),
                None,
                hess_times_grad_sdf_pts,
                eps=eps,
                rect=GRID_MAX
            )

            bending_term = (w * bending_loss_1).mean() + torch.sum(weights.view(-1) * bending_loss_2)

            del grad_pts, grad_pts_sq

        else:
            bending_loss = BENDING_FAKTOR * bendingEnergySDF(
                phasefield, None, hess_times_grad_sdf, eps=eps, rect=GRID_MAX
            ).view(-1, 1)

            bending_term = (w * bending_loss).mean()

        loss += bending_term

        # --- eikonal ---
        eikonal_loss = EIKONAL_FACTOR * eikonalEnergy(
            grad_sdf, eps=eps, rect=GRID_MAX
        ).view(-1, 1)

        eikonal_term = (w * eikonal_loss).mean()
        loss += eikonal_term

        # --- ambrosio-tortorelli ---
        at_loss = AMBROSIO_TORTORELLI_FACTOR * ambrosioTortorelli(
            phasefield, grad_phasefield, eps=eps, rect=GRID_MAX
        ).view(-1, 1)

        PF_term = (w * at_loss).mean()
        loss += PF_term

        # --- surface reconstruction ---
        pointcloud_loss = POINTCLOUD_FACTOR * pointcloudEnergy(
            sdf_pts, weight=weights
        )
        loss += pointcloud_loss

        # --- mass ---
        abs_sdf = torch.abs(sdf)
        abs_sdf2 = abs_sdf * abs_sdf
        abs_sdf3 = abs_sdf2 * abs_sdf

        if epoch <= 10:
            mass_loss = (0.001/eps)*MASS_FACTOR * (
                torch.exp(-100 * abs_sdf2) +
                3 * torch.exp(-100 * abs_sdf) +
                torch.exp(-10 * abs_sdf3)
            )
        elif epoch < 20:
            mass_loss = (0.001/eps)*MASS_FACTOR * (
                0.75 * torch.exp(-100 * abs_sdf2) +
                0.75 * 3 * torch.exp(-100 * abs_sdf) +
                torch.exp(-10 * abs_sdf3)
            )
        else:
            mass_loss = 0.5 * (0.001/eps)*MASS_FACTOR * (
                0.5 * torch.exp(-100 * abs_sdf2) +
                0.5 * 3 * torch.exp(-100 * abs_sdf) +
                torch.exp(-10 * abs_sdf3)
            )

        mass_loss += 0.001/eps * 250 * torch.exp(-15 * abs_sdf2)
        mass_term = (w * mass_loss.view(-1, 1)).mean()

        loss += mass_term

        # --- final ---
        loss = loss * 0.01  #scaling is only necessary to match the norm clipping
        loss.backward()

        # depending on training stage only step sdf_opt or both
        if self.train_sdf:
            torch.nn.utils.clip_grad_norm_(self._sdf_params, max_norm=75.0)
            self.opt_SDF.step()
            self.opt_SDF.zero_grad(set_to_none=True)


        if self.train_pf:
            torch.nn.utils.clip_grad_norm_(self._pf_params, max_norm=1.)
            self.opt_pf.step()
            self.opt_pf.zero_grad(set_to_none=True)
        
        # visualize intermediate meshes every 5 epochs using marching cubes
        if epoch % 5 == 0 and epoch > 0 and step % 500 == 0:
            mesh = imf2mesh(
                    lambda x: self.net_SDF(x), res=256, threshold=0., bound = 1.2, normalize = True, norm_type='res')
            if mesh is not None:
                mesh.export(osp.join(self.cfg.save_dir, "latest_mesh.obj")) 
        
            mesh_pf = imf2mesh(
                    lambda x: self.net_pf(x), res=128, threshold=0.5, bound = 1.2, normalize = True, norm_type='res')
            if mesh_pf is not None:
                mesh_pf.export(osp.join(self.cfg.save_dir, "latest_pf.obj"))      
        
        return {
            'loss': loss.detach().cpu().item(),
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/bending_loss': bending_term.detach().cpu().item(),
            'scalar/eikonal_loss': eikonal_term.detach().cpu().item(),
            'scalar/ambrosioTortorelli_loss': PF_term.detach().cpu().item(),
            'scalar/pointcloud_loss': pointcloud_loss.detach().cpu().item(),
            'scalar/mass_loss': mass_term.detach().cpu().item(),
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
        # validate is only used to compute adaptive grid
        def SDF_abs(x):
            return torch.abs(self.net_SDF(x))
        device = "cuda"
        GRID_MAX = tuple(cfg.trainer.training_params.grid_max_3d)
        update_grid_sdf = {2, 6, 7, 10, 15, 20, 25}
        update_grid_pf = {6, 7, 10, 15, 20, 25}
        if epoch in update_grid_sdf:
            if epoch in update_grid_pf:
                refined_midpoints, refined_levels = adaptive_refinement_3D(
                    box_midpoints=midpoints, 
                    refinement_lvl=lvl, 
                    functions=[SDF_abs, self.net_pf], 
                    threshold=[0.1, 0.75], 
                    device=device, 
                    dim=3
                )
                self.precompute_sampling_constants(
                refined_midpoints,
                refined_levels,
                cfg.input.parameters.bs,
                h = GRID_MAX[0] / cfg.sampling.init_grid_res,
                dim=3,
                device='cuda'
                )
            else: 
                refined_midpoints, refined_levels = adaptive_refinement_3D(
                    box_midpoints=midpoints, 
                    refinement_lvl=lvl, 
                    functions=[SDF_abs], 
                    threshold=[0.2], 
                    device=device, 
                    dim=3
                )
                self.precompute_sampling_constants(
                refined_midpoints,
                refined_levels,
                cfg.input.parameters.bs,
                h = GRID_MAX[0] / cfg.sampling.init_grid_res,
                dim=3,
                device='cuda'
                )
             
        else:
            refined_midpoints = midpoints
            refined_levels = lvl


        return {
            'loss': 0.0,
            'ref_mid': refined_midpoints,
            'levels':refined_levels} 



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

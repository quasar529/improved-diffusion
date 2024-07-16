import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import wandb
from tqdm import tqdm

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        t0=1000,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.t0 = t0

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model

        wandb.init(project="imporved_diffusion", entity="quasar529")
        wandb.config.update(
            {
                "batch_size": batch_size,
                "microbatch": microbatch,
                "lr": lr,
                "ema_rate": ema_rate,
                "log_interval": log_interval,
                "save_interval": save_interval,
                "resume_checkpoint": resume_checkpoint,
                "use_fp16": use_fp16,
                "fp16_scale_growth": fp16_scale_growth,
                "schedule_sampler": schedule_sampler,
                "weight_decay": weight_decay,
                "lr_anneal_steps": lr_anneal_steps,
                "t0": t0,
            }
        )
        wandb.run.name = f"diffusion_batch_{batch_size}_steps{lr_anneal_steps}_{self.resume_checkpoint}"

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev()))

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        total_steps = self.lr_anneal_steps if self.lr_anneal_steps else float("inf")
        with tqdm(total=total_steps, desc="Training Progress", dynamic_ncols=True) as pbar:
            while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
                batch, cond = next(self.data)
                self.run_step(batch, cond, self.t0)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                pbar.update(1)  # Update the progress bar

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, t0):
        self.forward_backward(batch, cond, t0)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond, t0):
        zero_grad(self.model_params)
        total_batches = (batch.shape[0] + self.microbatch - 1) // self.microbatch  # 전체 배치 수 계산
        progress_bar = tqdm(total=total_batches, desc="Processing Batches", leave=False, dynamic_ncols=True)

        for i in range(0, batch.shape[0], self.microbatch):
            current_batch = i // self.microbatch + 1  # 현재 배치 번호 계산
            progress_bar.set_description(
                f"Processing Batches {current_batch}/{total_batches}"
            )  # 진행 바 설명 업데이트

            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            t0_tensor = th.full((micro.shape[0],), t0, device=micro.device, dtype=th.long)
            noised_micro = self.diffusion.q_sample(micro, t0_tensor)  # 원본 이미지 x_start에 T0만큼의 노이즈 추가

            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                noised_micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            wandb.log({"train/loss": loss.item()})
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            if self.use_fp16:
                loss_scale = 2**self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)  # 배치 진행 상황 업데이트

        progress_bar.close()  # 배치 진행 바 종료

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2**self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad**2).sum().item()
            wandb.log({"train/grad_norm": np.sqrt(sqsum)})
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
            wandb.log({"train/lr": lr})

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        wandb.log({"train/step": self.step + self.resume_step})
        wandb.log({"train/samples": (self.step + self.resume_step + 1) * self.global_batch})
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self, save_dir="/home/jun/improved-diffusion/results"):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                save_path = bf.join(save_dir, filename)
                with bf.BlobFile(save_path, "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            opt_filename = f"opt{(self.step+self.resume_step):06d}.pt"
            opt_save_path = bf.join(save_dir, opt_filename)
            with bf.BlobFile(opt_save_path, "wb") as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(self.model.parameters(), master_params)
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

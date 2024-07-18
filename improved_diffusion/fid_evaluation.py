"""
Code from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/fid_evaluation.py
"""

import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm

from datetime import datetime


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=10000,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        self.samples_path = None  # 샘플 경로를 저장할 새 속성

        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    def set_samples_path(self, path):
        """생성된 샘플의 경로를 설정하는 메서드"""
        self.samples_path = path

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, f"{datetime.now().strftime('%Y%m%d_%H%M')}_dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(f"Stacking Inception features for {self.n_samples} samples from the real dataset.")
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.dl)
                except StopIteration:
                    break
                if isinstance(real_samples, (list, tuple)):
                    real_samples = real_samples[0]
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = torch.cat(stacked_real_features, dim=0).cpu().numpy()
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        # self.sampler.eval()
        # batches = num_to_groups(self.n_samples, self.batch_size)
        if self.samples_path is None:
            raise ValueError("Samples path is not set. Call set_samples_path() before calculating FID score.")

        # 저장된 샘플 로드
        samples_data = np.load(self.samples_path)
        if samples_data.files[0] == "arr_0":
            samples = samples_data["arr_0"]
        else:
            samples = samples_data[samples_data.files[0]]

        if len(samples) < self.n_samples:
            raise ValueError(f"Not enough samples. Found {len(samples)}, need {self.n_samples}")

        samples = samples[: self.n_samples]  # 필요한 만큼만 사용

        stacked_fake_features = []
        self.print_fn(f"Stacking Inception features for {self.n_samples} generated samples.")
        # for batch in tqdm(batches):
        for i in tqdm(range(0, self.n_samples, self.batch_size)):
            batch = samples[i : i + self.batch_size]
            batch = torch.from_numpy(batch).float().to(self.device) / 127.5 - 1  # 정규화
            batch = batch.permute(0, 3, 1, 2)  # NHWC to NCHW
            fake_features = self.calculate_inception_features(batch)
            # fake_samples = self.sampler.sample(batch_size=batch)
            # fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# Cosmos-NeMo-Assets example
custom_liquid_dataset = L(Dataset)(
    dataset_dir="datasets/benchmark_train/custom_liquid",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=custom_liquid_dataset,
    sampler=L(get_sampler)(dataset=custom_liquid_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_cosmos_nemo_assets
predict2_video2world_training_2b_custom_liquid = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="custom_liquid_1_6k",
    ),
    model=dict(
        config=dict(
            # Enable LoRA training
            train_architecture="lora",
            # LoRA configuration parameters
            lora_rank=16,
            lora_alpha=16,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            pipe_config=dict(
                ema=dict(enabled=True),     # enable EMA during training
                prompt_refiner_config=dict(enabled=False),  # disable prompt refiner during training
                guardrail_config=dict(enabled=False),   # disable guardrail during training
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=100000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-10),
    ),
    scheduler=dict(
        warm_up_steps=[1_000],
        cycle_lengths=[5_000],
        f_max=[0.6],
        f_min=[0.3],
    ),
)

for _item in [
    # 2b, cosmos_nemo_assets
    predict2_video2world_training_2b_custom_liquid,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

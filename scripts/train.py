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

import argparse
import importlib
import os

import attrs
import torch
import wandb
from loguru import logger as logging
from omegaconf import DictConfig, ListConfig, OmegaConf

from imaginaire.config import Config, pretty_print_overrides
from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed
from imaginaire.utils.config_helper import get_config_module, override


def config_to_dict(cfg):
    """
    Recursively converts a config object (including attrs classes and OmegaConf configs)
    into a dictionary that is serializable by wandb.
    It handles non-serializable types like torch.dtype by converting them to strings.
    This logic is adapted from imaginaire.lazy_config.lazy.LazyConfig.save_yaml
    """

    def is_serializable(item):
        try:
            # A quick check to see if OmegaConf can handle it.
            # This is a proxy for YAML/JSON serializability.
            OmegaConf.to_yaml(item)
            return True
        except Exception:
            return False

    def serialize_config(config):
        if isinstance(config, DictConfig):
            for key, value in config.items():
                if isinstance(value, (DictConfig, ListConfig)):
                    serialize_config(value)
                else:
                    if not is_serializable(value) and value is not None:
                        config[key] = str(value)
        elif isinstance(config, ListConfig):
            for i, item in enumerate(config):
                if isinstance(item, (DictConfig, ListConfig)):
                    serialize_config(item)
                else:
                    if not is_serializable(item) and item is not None:
                        config[i] = str(item)
        return config

    # 1. Convert the top-level attrs object to a dictionary.
    config_as_dict = attrs.asdict(cfg)
    # 2. Create an OmegaConf object, which can hold complex types.
    config_omegaconf = DictConfig(content=config_as_dict, flags={"allow_objects": True})
    # 3. Recursively serialize any non-standard objects to strings.
    serialized_config_omegaconf = serialize_config(config_omegaconf)
    # 4. Convert the sanitized OmegaConf object to a plain python dict.
    final_dict = OmegaConf.to_container(serialized_config_omegaconf, resolve=True)
    return final_dict


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    distributed.init()

    if distributed.is_rank0():
        wandb.init(
            entity='reasoner',
            project=config.job.project,
            group=config.job.group,
            name=config.job.name,
            config=config_to_dict(config),
            mode="offline", 
        )

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Create the model
    model = instantiate(config.model)
    # Create the dataloaders.
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiments=predict2_video2world_training_2b_cosmos_nemo_assets

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)

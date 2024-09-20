import os
import argparse
import logging
from pathlib import Path
import wandb
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from utils.read_config import generate_config
from utils.wandb_utils import init_or_resume_wandb_run
from pretrain.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from pretrain.lightning_trainer import LightningPretrain
from pretrain.lightning_datamodule import PretrainDataModule
from pretrain.lightning_trainer_spconv import LightningPretrainSpconv
from pytorch_lightning.callbacks import ModelCheckpoint

run = wandb.init(name='t1_name', config=None,
                            project='ReconsDistill',
                            entity='trailab',
                            group='t1_group',
                            job_type='t1_job', dir='/slidr/output')


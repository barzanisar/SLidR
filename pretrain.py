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

pl.seed_everything(42, workers=True)

def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="num gpus"
    )
    parser.add_argument(
        "--extra_tag", type=str, default=None, help="pretrain extra tag"
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file)

    if args.extra_tag:
        config['extra_tag'] = args.extra_tag
    if args.num_gpus > 1:
         config['num_gpus'] = args.num_gpus
    

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    print(f"Creating output directories: ")
    savedir_root = os.path.join(config["save_dir"], "Pretraining", config['desc'], config["extra_tag"])
    if os.environ.get("LOCAL_RANK", 0) == 0:
        os.makedirs(savedir_root, exist_ok=True)
    checkpoints_dir = os.path.join(savedir_root, 'checkpoints')
    config['savedir_root'] = savedir_root
    print(f"Savedir_root: {savedir_root}")
    print(f"checkpoints_dir: {checkpoints_dir}")
    
    wandb_logger = None
    if config.get("wandb", {}).get("enabled"):
        run_id_file = Path(savedir_root)/ 'wandb_run_id.txt'
        config['wandb']['group'] = config['extra_tag']
        config['wandb']['job_type'] = 'pretrain'
        init_or_resume_wandb_run(run_id_file, config)
        wandb_logger = pl_loggers.WandbLogger(experiment=wandb.run)
        wandb_logger.log_dir = savedir_root

    if args.resume_path:
        resume_path = args.resume_path
    elif config["resume_path"]:
        # Resume either from config or last saved checkpoint
        resume_path = config["resume_path"]
    else:
        resume_path = os.path.join(checkpoints_dir, 'last.ckpt')
        if not os.path.isfile(resume_path):
            resume_path=None
    
    config["resume_path"] = resume_path
    logging.info(f"Ckpt resume_path: {resume_path}")

    dm = PretrainDataModule(config)
    model_points, model_images = make_model(config)
    if config["num_gpus"] > 1:
        model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
    if config["model_points"] == "minkunet":
        module = LightningPretrain(model_points, model_images, config)
    elif config["model_points"] == "voxelnet":
        module = LightningPretrainSpconv(model_points, model_images, config)
    #path = os.path.join(config["working_dir"], config["datetime"])
    # ckpt_path=os.path.join(path, 'ckpts')

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch:02d}",
        every_n_epochs=0,
        save_top_k=-1,
        save_last=True
    )

    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        accelerator="cuda",
        default_root_dir=savedir_root,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=0,
        resume_from_checkpoint=resume_path,
        logger=wandb_logger,
        max_epochs=25,
        check_val_every_n_epoch=5,
        log_every_n_steps=100, deterministic=True,
        replace_sampler_ddp=False

    )
    print("Starting the training")
    trainer.fit(module, dm) #, ckpt_path=resume_path


if __name__ == "__main__":
    main()

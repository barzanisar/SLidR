import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pytorch_lightning.utilities import rank_zero_only


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        # self.epoch = 0
        # if config["resume_path"] is not None:
        #     self.epoch = int(
        #         re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
        #     )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        # self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        # if os.environ.get("LOCAL_RANK", 0) == 0:
        #     os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F #(num voxels in batch, C=64)
        self.model_images.eval()
        self.model_images.decoder.train()
        output_images = self.model_images(batch["input_I"]) #(6 views*4 bs=24, C=64, H=224, W=416)

        del batch["sinput_F"]
        del batch["sinput_C"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))

        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]] #[num pts=4096, 64] pt features
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m] #[num pixels=4096, 64] corresponding img pixel features
        return self.criterion(k, q)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        # computes avg features for points and pixels for each segment (i.e. computes superpoint and superpixel features and matching/pairing point and img features)
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"] #(24, 224, 416) slic segment ids
        pairing_images = batch["pairing_images"] #(num pairing pts in batch, 3= [0 to 23 image id, v pixel, u pixel])
        pairing_points = batch["pairing_points"] #(num pairing pts in batch, inverse_indices)

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size, #24 images * 150 superpixels
                self.superpixel_size, # 150
                device=self.device,
            )[:, None, None] + superpixels # (24,1,1) [0, 150, 300, 450, ...] + (24, 224, 416) slic segment ids
        ) # to increase segment ids for each img by 150
        m = tuple(pairing_images.cpu().T.long()) #tuple: 0: img ids, 1: v pixels, 2: u pixels

        superpixels_I = superpixels.flatten() #(24, H, W) -> (24*H*W)
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device) # 0, ..., num pairing pts -1  
        total_pixels = superpixels_I.shape[0] # num pixels in all 24 imgs
        idx_I = torch.arange(total_pixels, device=superpixels.device) # 0, ..., num pixels-1

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            ) # sparse one hot matrix of size: (24 imgs *150 max segments, num FOV points for all 24 imgs) where first input is (row indices: segment/superpixel ids of pairing pts, col indices: torch arange num pairing pts), second input is values i.e. ones to put in those row and col indices

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            ) # sparse one hot matrix of size: (24 imgs *150 max segments, total pixels in all 24 imgs)

        #each col has only one 1 i.e. each pairing pt (one hot P) and pixel (one hot I) can have only one segment id
        # However, each row i.e. each segment can have multiple pairing pts and their pixels
        # k is the segment-wise feature from the point backbone = average features of all pairing pts in this segment 
        # k = [num segments,  num pairing pts] @ [num pairing pts, 64=feature dim]
        # output_points is voxel-wise features (num voxels, feature dim) -> output_points[pairing_points contains voxel index for each pairing pt] gives pairing point-wise features 
        k = one_hot_P @ output_points[pairing_points] #gives sum of all pairing pt features in a segment i.e. [num segments, 64]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6) #[num segments=24*150=3600, 64] / [num segments, 1]


        # q = segment-wise output image features = avg features of all pixels in the segment
        # q =  [num segments,  num all pixels] @ [num all pixels, 64=feature dim] -> sums pixel-wise features across all pixels in a segment to get segment wise features
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2) # q = [num segments, 64]
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6) # #[num segments, 64] / [num segments, 1] i.e. divide by num pixels in a segment to get avg segment feature from the image backbone

        mask = torch.where(k[:, 0] != 0) # since k has features for 150 segments in each image, not all segment ids are present in an img so their feature vectors will be zero. Hence, select non-zero segment features
        k = k[mask]
        q = q[mask]

        return self.criterion(k, q)

    def training_epoch_end(self, outputs):
        # self.epoch += 1
        if self.current_epoch == self.num_epochs-1:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        self.model_images.eval()
        output_images = self.model_images(batch["input_I"])

        losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self._config['savedir_root'], "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.current_epoch,
                "config": self._config,
            },
            path,
        )

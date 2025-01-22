from defect_detection import AE_cls, deepAE_load
import lightning as L
import torch
from torchvision.utils import save_image

class ModelWrapper(L.LightningModule):
    """A PyTorch Lightning wrapper for the autoencoder models.
    """
    def __init__(self, name="deepAE", model_cfg_path="AE_model"):
        super().__init__()
        # self.model = AE_cls()
        # if name == "deepAE" and model_cfg_path!="":
        #     self.model = deepAE_load(model_cfg_path+"/")
        self.model = deepAE_load("AE_model/")

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.model(x)
        lossfunc = torch.nn.MSELoss() #This should be configured in the config file
        loss = lossfunc(x_hat, x)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.model(x)
        lossfunc = torch.nn.MSELoss()
        loss = lossfunc(x_hat, x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log the first input and output images of the validation set
        if batch_idx==0:
            for xb, io in zip([x, x_hat], ["input", "output"]):
                if io=='input' and self.current_epoch>1:
                    continue # Skip logging input images after the first epoch
                imgname = f"images/val_{io}_sample.png"
                save_image(xb[0], imgname)
                self.logger.experiment.log_image(imgname)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
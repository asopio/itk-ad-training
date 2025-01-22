from lightning.pytorch.cli import LightningCLI
# from lightning.loggers import CometLogger
# from defect_detection import *  # This will import all your models
from module import ModelWrapper
from data_module import DefectDataModule

def main():
    cli = LightningCLI(ModelWrapper, DefectDataModule, 
                       parser_kwargs={"fit":{"default_config_files": ["configs/default_config.yaml"]}},
                       save_config_kwargs={"overwrite": True}  )


if __name__ == "__main__":
    main() 
import pytorch_lightning as pl,torch,yaml,argparse
from pytorch_lightning import Trainer,logging
from utils import tools
from model import prototype

#todo ---------------------------------------Init Model----------------------------------------
config_file = 'config/config.yaml'
config = tools.get_config(config_file)
model = prototype.Prototype(config)

#todo --------------------------------------Init Trainer---------------------------------------
#init logger
logger = logging.TestTubeLogger(
	save_dir=config['save_dir'],
	version=0,
    name=config['prototype']
)

trainer = Trainer(
    logger=logger,
    distributed_backend='ddp2',
    gpus=config['gpu_ids'],
    track_grad_norm=2,
    print_nan_grads=True,
    show_progress_bar=True,
    log_save_interval=100,
    train_percent_check=0.01,
    max_nb_epochs=config['epoch']
)



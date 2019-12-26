import pytorch_lightning as pl
from pytorch_lightning import Trainer,logging
from pytorch_lightning.callbacks import ModelCheckpoint
import template,os


model = template.Template()

#todo checkpoint
# edit checkpoint
checkpoint_callback = ModelCheckpoint(
	filepath= os.getcwd(),
	save_best_only=True,
	mode='min',
	verbose=True,
	save_weights_only=True,
	period=1,
	prefix=''
)
trainer = Trainer(checkpoint_callback=checkpoint_callback)
#restoring training session
logger = TestTubeLogger(
	save_dir='./savepath',
	version=1
)

trainer = Trainer(logger=logger,
                  default_save_path='./savepath'
                  )
#this call loads model weights and trainer state
#the trainer restores global_step ,current_epoch, all optimizer, all lr_schedulers, model weights
#the trainer continues samlessly from where you left off
trainer.fit(model)

#todo distributed training
#choosing a backend from
# DataParallel(DP),
# DistributedDataParallel(DDP) training a copy on each gpu only syncs gradients
#DDP-2, except each node trains a single copy of the model using ALL GPUs on that node.

#Default(single GPU or CPU)
trainer = Trainer(distributed_backend=None)
#DP(gpus>1)
trainer = Trainer(distributed_backend='dp')
#DDP
trainer = Trainer(distributed_backend='ddp')
#DDP2
trainer = Trainer(distributed_backend='ddp2')

#specify which GPUs to use
#how many gpus
Trainer(gpus=2)
#which gpus
Trainer(gpus=[0,1])
Trainer(gpus='0,1')
Trainer(gpus=-1)#this will use all available GPUs

#mix of set gpu
trainer = Trainer(gpus=4,nb_gpu_nodes=1,distributed_backend='ddp2')

#todo logging
#default_save_path
Trainer(default_save_path='.')
#set logger
logger = logging.TestTubeLogger('./haha')
trainer = Trainer(logger = logger)#this will overwrite default_save_path

#choose from several types of logger
#test tubu (subclass of PyTorch SummaryWritter
from pytorch_lightning.logging import TestTubeLogger,MLFlowLogger,CometLogger
tt_logger = TestTubeLogger(
	save_dir='.',
	name = 'default',
	debug=False,
	create_git_tag=False
)
#MLFlow
mlf_logger = MLFlowLogger(
	experiment_name='default',
	tracking_uri='file:/.',
)

#Custom logger
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

class Mylogger(LightningLoggerBase):
	@rank_zero_only
	def log_hyperparams(self, params):
		pass
	@rank_zero_only
	def log_metrics(self, metrics, step_num):
		pass

	def save(self):
		pass

	@rank_zero_only
	def finalize(self, status):
		pass

#display metrics in progress bar
trainer = Trainer(show_progress_bar=True)

#log metric row every k batches
trainer = Trainer(row_log_interval=10)#save a .csv log file every 10 batches

#log gpu memory
#default
trainer = Trainer(log_gpu_memory=None)
#min/max utilization
trainer = Trainer(log_gpu_memory='min_max')
#log all the GPU mrmory (if on DDP, log only that node)
trainer = Trainer(log_gpu_memory='all')

#process position
#decide which progress bar to use
trainer = Trainer(process_position=0)

#save all hyperparameters
#automatically log hyperparameteres stored in the hparams as an argparse.Namespace
class MyModel(pl.LightningModule):
	def __init__(self,hparams):
		super(MyModel, self).__init__()
		self.hparams = hparams
args = {}
model = MyModel(args)

logger  = TestTubeLogger(save_dir='.')
t = Trainer(logger=logger)

#write logs file to csv every k batches
trainer = Trainer(log_save_interval=100)


#todo training loop
#accumulate gradients
trainer = Trainer(accumulate_grad_batches=1)
#training for min or max epochs
trianer = Trainer(min_nb_epochs=1,max_nb_epochs=1000)
#early stopping
from pytorch_lightning.callbacks import EarlyStopping
early_stop_callback = EarlyStopping(
	monitor='val_loss',
	min_delta=0.00,
	patience=100,
	verbose=False,
	mode='min'
)

trainer = Trainer(early_stop_callback = early_stop_callback)
#pass in None to disable it
trainer = Trainer(early_stop_callback = None)

#Gradient clipping
trainer = Trainer(gradient_clip_val=0.5)
#inspect gradient norm
#-1 does not track, other positive integer n track the LP norm(P=2here)
trainer = Trainer(track_grad_norm=-1)
trainer = Trainer(track_grad_norm=2)

#set how much of the training set to check
trainer = Trainer(train_percent_check=0.1)

#truncated back propagation through time
#this flag enables each batch split into sequences of size truncated_bptt_steps and passed to training_step()

#default
trainer = Trainer(truncated_bptt_steps=None)
#split
trainer = Trainer(truncated_bptt_steps=2)


#todo debugging

#fast dev run(run with 1training and 1 validation batch)
trainer = Trainer(fast_dev_run=False)#default

#overfit on subset of data
trainer = Trainer(overfit_pct=0.01)#pct = overfit percent check

#print the parameter count by layer
trainer = Trainer(weights_summary='top')# only print the top_level modules

#print which gradients are nan
trainer = Trainer(print_nan_grads=True)




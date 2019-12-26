import cv2
from pytorch_lightning import logging
import pytorch_lightning as pl

logger = logging.TestTubeLogger('.')
logger.experiment.add_scalar()

trainer = pl.Trainer()
trainer.fit()
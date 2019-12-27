import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
import pytorch_lightning as pl

#todo -----------------------------------------import usr's module----------------------------------
from .networks import CoarseGenerator,FineGenerator
import model.networks as NWK
from utils import tools
from data import dataset

class Prototype(pl.LightningModule):

    def __init__(self,config):
        super(Prototype, self).__init__()
        self.input_dim = config['input_dim']
        self.config = config
        # not the best model...
        self.generator = Generator(config['netG'])
        self.coarse_gen_num = config['coarse_gen_num']
        #build coarse discriminator
        coarse_dis_list = []
        for i in range(self.coarse_gen_num):
            coarse_dis_list.append(NWK.LocalDis(config['netD']))
        self.coarse_dis_cluster = nn.Sequential(*coarse_dis_list)
        #final discriminator
        self.global_dis = NWK.GlobalDis(config['netD'])
        self.local_dis = NWK.LocalDis(config['netD'])


    def forward(self,x,  bboxes, masks,):

        x1, x2, offset_flow = self.netG(x, masks)
        x1_inpaints = x1 * masks.repeat(1,self.coarse_gen_num,1,1) + x * (1. - masks.repeat(1,self.coarse_gen_num,1,1))
        x2_inpaint = x2 * masks + x * (1. - masks)
        local_patch_x1_inpaint = tools.local_patch(x1_inpaints,bboxes)
        local_patch_x2_inpaint = tools.local_patch(x2_inpaint, bboxes)
        return x1,x2,local_patch_x1_inpaint,local_patch_x2_inpaint

    def training_step(self, batch, batch_nb,optimizer_idx):
        #this only collect a piece of data 1/ngpus
        # REQUIRED
        bboxes = tools.random_bbox(self.config, batch_size=batch.size(0))
        x, mask = tools.mask_image(batch, bboxes, self.config)
        local_patch_gt = tools.local_patch(batch, bboxes)
        x1,x2,local_x1,local_x2 = self.forward(x,bboxes,mask)
        def dis_forward(model,pos,neg):
            bs = pos.size(0)
            assert pos.size(0)==neg.size(0)
            pos_pred,neg_pred = model(torch.cat([pos,neg],dim=0)).split(bs,dim=0)
            return pos_pred,neg_pred
        if optimizer_idx==0 and batch_nb%self.config['train_generator_interval']==0:
            #reconstruction_loss
            l1_loss = nn.L1Loss()
            sd_mask = tools.spatial_discounting_mask(self.config)
            sd_loss = l1_loss(local_x1 * sd_mask.repeat(1,self.coarse_gen_num,1,1), local_patch_gt * sd_mask) * \
                           self.config['coarse_l1_alpha'] + \
                           l1_loss(local_x2 * sd_mask, local_patch_gt * sd_mask)
            self.logger.experiment.add_scalar('sd_loss',sd_loss,self.global_step)
            ae_loss = l1_loss(x1 * (1. - mask).repeat(1,self.coarse_gen_num,1,1), batch* (1. - mask)) * \
                           self.config['coarse_l1_alpha'] + \
                           l1_loss(x2 * (1. - mask), batch * (1. - mask))
            self.logger.experiment.add_scalar('ae_loss',ae_loss,self.global_step)
            #dis_losses
            local_x2_pred = self.local_dis(local_x2).mean()
            global_x2_pred = self.global_dis(x2).mean()
            self.logger.experiment.add_scalar('local_dis_loss',-local_x2_pred,self.global_step)
            self.logger.experiment.add_scalar('global_dis_loss',-global_x2_pred,self.global_step)
            local_x1s = local_x1.split(self.input_dim,dim=1)
            local_x1_pred = []
            for local_x1_i,coarse_dis in zip(local_x1s,self.coarse_dis_cluster):
                local_x1_pred.append(coarse_dis(local_x1_i)).mean()
            local_x1_pred = torch.cat(local_x1_pred,dim=0).mean()
            self.logger.experiment.add_scalar('coarse_dis_loss', local_x1_pred/self.coarse_gen_num, self.global_step)
            output = {
                'loss':ae_loss+sd_loss-local_x2_pred-global_x2_pred+local_x1_pred,
                'progress_bar':{'sd_loss':sd_loss,'ae_loss':ae_loss,
                                'loca_dis_loss':-local_x2_pred,'global_dis_loss':-global_x2_pred,
                                'coarse_dis_loss':local_x1_pred/self.coarse_gen_num}
            }
            return output

        if optimizer_idx==1 and batch_nb%self.config['train_coarse_discriminator_interval']==0:
            #get coarse discriminator loss
            local_x1s = list(local_x1.split(self.input_dim, dim=1))
            coarse_dis_loss = 0
            for i,_ in enumerate(zip(local_x1s,self.coarse_dis_cluster)):
                local_x1_i, coarse_dis = _
                if i==0:
                    pos_sample = torch.cat(local_x1s[1:],dim=0).detach()
                elif i==self.coarse_gen_num-1:
                    pos_sample = torch.cat(local_x1s[:-1],dim=0).detach()
                else:
                    pos_sample = torch.cat(local_x1s[:i-1]+local_x1s[i+1:],dim=0).detach()
                neg_sample = local_x1_i.repeat(self.coarse_gen_num,1,1,1).detach()

                pos_pred,neg_pred = dis_forward(coarse_dis,pos_sample,neg_sample)
                coarse_dis_loss+=neg_pred.mean()-pos_pred.mean()

            return {'loss':coarse_dis_loss}

        if optimizer_idx==2 and batch_nb%self.config['train_final_discriminator_interval']==0:
            loss = 0
            #global dis
            pos_sample = batch.detach()
            neg_sample = x2.detach()
            pos_pred,neg_pred = dis_forward(self.global_dis,pos_sample,neg_sample)
            loss+=-pos_pred.mean()+neg_pred.mean()
            pos_sample = local_patch_gt.detach()
            neg_sample = local_x2.detach()
            pos_pred,neg_pred = dis_forward(self.local_dis,pos_sample,neg_sample)
            loss+=-pos_pred.mean()+neg_pred.mean()
            return {'loss':loss}

    def configure_optimizers(self):
        # optimizer
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        optimizer_d = torch.optim.Adam(d_params, lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        optimizer_cd = torch.optim.Adam(self.coarse_dis_cluster.parameters(), lr=self.config['lr'],
                                             betas=(self.config['beta1'], self.config['beta2']))
        return optimizer_g,optimizer_cd,optimizer_d
    @pl.data_loader
    def train_dataloader(self):
        dst = dataset.Dataset(self.config['data_path'],
                              self.config['image_shape'],
                              self.config['with_subfolder'],
                              random_crop=self.config['random_crop'])
        return DataLoader(dst,num_workers=self.config['num_workers'],shuffle=True, batch_size=self.config['batch_size'])

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.coarse_gen_num = config['coarse_gen_num']
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']

        # build coarse_generator cluster
        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum)
        self.coarse_generator.conv17 = NWK.gen_conv(self.cnum//2,self.input_dim*self.coarse_gen_num,3,1,1,activation='none')
        # build fine_generator
        self.fine_generator = FineGenerator(self.input_dim * self.coarse_gen_num, self.cnum)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x,mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow
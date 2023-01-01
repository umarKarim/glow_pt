import argparse 
import os 
from shutil import rmtree 
import time  
import torch

from dataset import get_dataloader 
from trainer import Trainer 

class Train():
    def __init__(self, opts):
        self.experiments_dir = opts.experiments_dir
        self.dataset_path = opts.dataset_path 
        self.nbits = opts.nbits 
        self.im_size = opts.im_size  
        self.gpu_ids = opts.gpu_ids
        self.train_epochs = opts.train_epochs 
        self.tb_dir = opts.tb_dir 
        self.console_iter = opts.console_iter
        self.tb_iter = opts.tb_iter 
        self.save_iter = opts.save_iter 
        self.dataset_name = opts.dataset 
        self.batch_size = opts.batch_size 
        self.lr = opts.lr 
        self.pretrain_path = opts.pretrain_path 
        self.num_blocks = opts.num_blocks 
        self.num_flows = opts.num_flows 

        os.makedirs(self.experiments_dir, exist_ok=True)

        # getting the dataloader 
        self.dataloader = get_dataloader(name=self.dataset_name,
                                        batch_size=self.batch_size,
                                        nbits=self.nbits,
                                        path=self.dataset_path,
                                        im_size=self.im_size)
        
        self.trainer = Trainer(
            im_size=self.im_size,
            experiments_dir=self.experiments_dir,
            gpu_ids=self.gpu_ids,
            tb_dir=self.tb_dir, 
            lr=self.lr,
            pretrain_path=self.pretrain_path,
            num_blocks=self.num_blocks,
            num_flows=self.num_flows,
            nbits=self.nbits
        )

        self.global_iter = 0
        self.st_time = 0

    def start_training(self):
        self.st_time = time.time() 
        for ep in range(self.train_epochs):
            for i, data in enumerate(self.dataloader):
                if i == 0:
                    with torch.no_grad():
                        self.trainer.forward_step(data)
                        continue
                self.trainer.forward_step(data)
                self.trainer.compute_loss()
                self.trainer.optim_step()
                if self.global_iter % self.console_iter == 0:
                    self.trainer.console_display(
                        epoch=ep,
                        batch_count=i, 
                        elapsed_time=time.time() - self.st_time
                    )
                if self.global_iter % self.tb_iter == 0:
                    self.trainer.update_tb(self.global_iter)
                if self.global_iter % self.save_iter == 0 and self.global_iter > 0:
                    self.trainer.save_model(self.global_iter)
                self.global_iter += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', type=str, default='glow_20221213/')
    parser.add_argument('--dataset_path', type=str, default='/datasets/celebA/img_align_celebA/')
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--nbits', type=int, default=5)
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--tb_dir', type=str, default='runs/')
    parser.add_argument('--console_iter', type=int, default=50)
    parser.add_argument('--tb_iter', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default='5000')
    parser.add_argument('--dataset', type=str, default='celebA')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_flows', type=int, default=32)
    parser.add_argument('--pretrain_path', type=str, default=None)
    opts = parser.parse_args()
    Train(opts).start_training()
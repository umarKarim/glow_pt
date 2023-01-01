import torch 
import torch.nn as nn
import torchvision  
from torch.utils.tensorboard import SummaryWriter 
import os 
import shutil 

from network import Network 



class Trainer():
    def __init__(self, im_size, experiments_dir, gpu_ids, tb_dir, lr, pretrain_path=None, 
                num_blocks=4, num_flows=32, nbits=5):
        self.im_size = im_size 
        self.experiments_dir = experiments_dir 
        self.gpu_ids = gpu_ids
        self.tb_dir = tb_dir
        self.lr = lr 
        self.pretrain_path = pretrain_path
        self.num_blocks = num_blocks 
        self.num_flows = num_flows  
        self.nbits = nbits 

        # setting up the directories 
        self.model_dir = os.path.join(self.experiments_dir, 'saved_models/')
        os.makedirs(self.model_dir, exist_ok=True)

        # tensorboard 
        if os.path.isdir(self.tb_dir):
            shutil.rmtree(self.tb_dir)    
        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        # device 
        if torch.cuda.is_available() and len(self.gpu_ids) > 0:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # network 
        self.net = Network(num_blocks=self.num_blocks, num_flows=self.num_flows)
        if self.pretrain_path is not None:
            self.net.load_state_dict(torch.load(self.pretrain_path))
        if len(self.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, self.gpu_ids)
        self.net = self.net.to(self.device)
        # optimizer 
        self.optim = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                betas=(0.9, 0.99))

        # setting up z for tensorboard 
        self.z = self.z_for_tb()

        # setting up the input and output data for training 
        self.in_image = None
        self.in_noisy = None 
        self.log_p = None 
        self.log_det = None 
        self.loss = None 
    
    def forward_step(self, data):
        self.in_image = data 
        self.in_noisy = self.in_image + torch.rand_like(self.in_image) / (2 ** self.nbits)
        self.in_noisy = self.in_noisy.to(self.device)
        log_p, log_det, _ = self.net(self.in_noisy)
        self.log_p = log_p.mean()
        self.log_det = log_det.mean()
        
    def compute_loss(self):
        tot_pixels = self.im_size * self.im_size * 3
        quant_loss = -torch.log(torch.tensor(2 ** self.nbits)) * tot_pixels 
        loss = self.log_p + self.log_det + quant_loss
        self.loss = (-loss / (torch.log(torch.tensor(2)) * tot_pixels)).mean()

    def optim_step(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

    def console_display(self, epoch, batch_count, elapsed_time):
        print('-' * 30)
        tot_pixels = self.im_size * self.im_size * 3 
        log_p = (self.log_p / (torch.log(torch.tensor(2)) * tot_pixels)).mean()
        log_det = (self.log_det / (torch.log(torch.tensor(2)) * tot_pixels)).mean()
        print('Epoch {}, batch {}, time {}'.format(epoch, batch_count, elapsed_time))
        print('log_p {}, log_det {}, tot_loss {}'.format(log_p, log_det, self.loss))
        print('-' * 30)

    def update_tb(self, global_iter):
        # the losses 
        tot_pixels = self.im_size * self.im_size * 3
        all_losses = {'log_p': (self.log_p / (torch.log(torch.tensor(2)) * tot_pixels)).mean(),
                        'log_det': (self.log_det / (torch.log(torch.tensor(2)) * tot_pixels)).mean(),
                        'tot_loss': self.loss}
        self.tb_writer.add_scalars('Lossses', all_losses, global_step=global_iter)
        # the images 
        with torch.no_grad():
            try:
                ims = self.net.reverse(self.z)
            except AttributeError:
                ims = self.net.module.reverse(self.z)
        ims = torch.clip(ims, -0.5, 0.5)
        ims = torchvision.utils.make_grid(ims, normalize=True, value_range=(-0.5, 0.5))
        self.tb_writer.add_image('Generated ims', ims, global_step=global_iter)    

    def z_for_tb(self):
        z_shapes = self.calc_z_shapes(3, self.im_size, self.num_flows, self.num_blocks)
        z_sample = [] 
        for z in z_shapes:
            z_new = torch.randn(20, *z) * 0.7
            z_sample.append(z_new.to(self.device))
        return z_sample 

    def calc_z_shapes(self, n_channel, input_size, num_flows, num_blocks):
        z_shapes = []
        for i in range(num_blocks - 1):
            input_size //= 2
            n_channel *= 2
            z_shapes.append((n_channel, input_size, input_size))
        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))
        return z_shapes

    def save_model(self, global_iter):
        model_name = ('%010d.pth' % global_iter) 
        model_path = os.path.join(self.model_dir, model_name)
        try: 
            torch.save(self.net.module.state_dict(), model_path)
        except:
            torch.save(self.net.state_dict(), model_path)
        print('=' * 30)
        print('MODEL SAVED at {}'.format(model_path))
        print('=' * 30)


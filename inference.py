import argparse 
import torch 
import numpy as np  
import matplotlib.pyplot as plt 

from network import Network 


class Inference():
    def __init__(self, model_path, temp, num_blocks, num_flows, im_size,
                save_path):
        self.model_path = model_path 
        self.temp = temp 
        self.num_blocks = num_blocks
        self.num_flows = num_flows 
        self.im_size = im_size 
        self.save_path = save_path 

        self.net = Network(num_blocks=self.num_blocks, 
                           num_flows=self.num_flows, 
                           inf=True)
        self.net.load_state_dict(torch.load(self.model_path, 
                                            map_location=torch.device('cpu')))
        self.z = self.get_z()
        with torch.no_grad():
            try:
                self.im = self.net.reverse(self.z)
            except AttributeError:
                self.im = self.net.module.reverse(self.z)
        self.np_im = self.get_np_im()
        self.save_im()
        self.show_im()

    def get_z(self):
        z_shapes = self.calc_z_shapes(3, self.im_size, self.num_flows, self.num_blocks)
        z_sample = [] 
        for z in z_shapes:
            z_new = torch.randn(1, *z) * self.temp 
            z_sample.append(z_new)
        return z_sample

    def calc_z_shapes(self, n_channel, input_size, num_flows, num_blocks):
        z_shapes = []
        for _ in range(num_blocks - 1):
            input_size //= 2
            n_channel *= 2
            z_shapes.append((n_channel, input_size, input_size))
        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))
        return z_shapes

    def get_np_im(self):
        im = self.im.squeeze().cpu().numpy() + 0.5
        im = np.transpose(im, (1, 2, 0))
        im = np.clip((im * 255.0), 0.0, 255.0).astype(np.uint8)
        return im 

    def show_im(self):
        plt.imshow(self.np_im)
        plt.show()

    def save_im(self):
        plt.imsave(self.save_path, self.np_im)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='glow_20221213/saved_models/0000070000.pth')
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--num_flows', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='inf_result.png')
    args = parser.parse_args()
    Inference(model_path=args.model_path,
              temp=args.temp,
              num_blocks=args.num_blocks,
              num_flows=args.num_flows,
              im_size=args.im_size,
              save_path=args.save_path)
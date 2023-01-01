import torch 
import torch.nn as nn
import torch.nn.functional as F  
from math import pi, log 


class ActNorm(nn.Module):
    def __init__(self, ch=3, inf=False):
        super().__init__()
        self.ch = ch 
        self.scale = nn.Parameter(torch.ones(1, self.ch, 1, 1))
        self.loc = nn.Parameter(torch.zeros(1, self.ch, 1, 1))
        self.inf = inf 
        if self.inf:
            self.init = True 
        else:
            self.init = False 

    def initialize(self, x):
        b, c, h, w = x.shape 
        flat = x.permute(1, 0, 2, 3).contiguous().view(c, -1)
        mean = flat.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
        std = flat.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))
        
    def forward(self, x):
        if not self.init:
            self.initialize(x)
            self.init = True
        y = self.scale * (x + self.loc) 
        b, _, h, w = x.shape 
        log_s = torch.log(torch.abs(self.scale))
        log_det = h * w * log_s.sum() 
        return y, log_det 

    def reverse(self, y):
        x = y / self.scale - self.loc 
        return x 

class InvConv2d(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.ch = ch 
        w = torch.randn((self.ch, self.ch))
        q, _ = torch.linalg.qr(w)
        w = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        _, _, h, w = x.shape 
        y = F.conv2d(x, self.weight)
        log_det = h * w * torch.slogdet(self.weight.squeeze().double())[1].float()
        return y, log_det

    def reverse(self, y):
        weight_inv = self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        x = F.conv2d(y, weight_inv)
        return x


class ScaledConv(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.conv = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros((1, self.out_ch, 1, 1)))

    def forward(self, x):
        out = F.pad(x, [1, 1, 1, 1], value=1)
        out = self.conv(out) 
        out = out * torch.exp(self.scale * 3)
        return out
        

class AffineCoupling(nn.Module):
    def __init__(self, ch=3, filters=512):
        super().__init__()
        self.ch = ch 
        self.filters = filters 

        self.net = nn.Sequential(
            nn.Conv2d(self.ch // 2, self.filters, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.filters, self.filters, 1),
            nn.ReLU(),
            ScaledConv(self.filters, self.ch // 2)
        )
        self.net[0].weight.data.normal_(0.0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0.0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        xa, xb = x.chunk(2, 1)
        ya = xa 
        out = self.net(xa)
        yb = xb + out 
        y = torch.cat((ya, yb), dim=1)
        logdet = 0
        return y, logdet 

    def reverse(self, y):
        ya, yb = y.chunk(2, 1)
        out = self.net(ya)
        xb = yb - out
        xa = ya 
        x = torch.cat((xa, xb), dim=1)
        return x 



class Flow(nn.Module):
    def __init__(self, ch=3, inf=False):
        super().__init__()
        self.ch = ch 
        self.inf = inf 
        self.actnorm = ActNorm(self.ch, inf=self.inf)
        self.invconv = InvConv2d(self.ch)
        self.coupling = AffineCoupling(self.ch)

    def forward(self, x):
        out, log_det1 = self.actnorm(x)
        out, log_det2 = self.invconv(out)
        out, log_det3 = self.coupling(out)
        log_det = log_det1 + log_det2 + log_det3 
        return out, log_det

    def reverse(self, x):
        out = self.coupling.reverse(x)
        out = self.invconv.reverse(out)
        out = self.actnorm.reverse(out)
        return out



class Block(nn.Module):
    def __init__(self, ch=3, num_flows=32, split=True, inf=False):
        super().__init__()
        self.num_flows = num_flows 
        self.ch = ch 
        self.split = split 
        
        # for the multiscale arch 
        if self.split:
            self.prior = ScaledConv(in_ch=ch * 2, out_ch=ch * 4)
        else:
            self.prior = ScaledConv(in_ch=ch * 4, out_ch=ch * 8)
        
        # the flows in the network 
        self.flows = nn.ModuleList()
        for _ in range(self.num_flows):
            self.flows.append(Flow(ch=self.ch * 4, inf=inf))

    def forward(self, x):
        b, c, h, w = x.shape 
        # squeezing the input 
        x_sq = x.view(b, c, h // 2, 2, w // 2, 2)
        x_sq = x_sq.permute(0, 1, 3, 5, 2, 4)
        x_sq = x_sq.contiguous().view(b, c * 4, h // 2, w // 2)
        out = x_sq 
        # the flows 
        log_det = 0 
        for flow in self.flows:
            out, det = flow(out)
            log_det += det 
        # splitting the output if required 
        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = self.gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b, -1).sum(1)
        else:
            zeros = torch.zeros_like(out)
            mean, log_sd = self.prior(zeros).chunk(2, 1)
            log_p = self.gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b, -1).sum(1)
            z_new = out 
        return out, log_det, log_p, z_new 

    def reverse(self, input, z_in):
        if self.split:
            mean, log_sd = self.prior(input).chunk(2, 1)
            z = self.gaussian_sample(z_in, mean, log_sd)
            input = torch.cat([input, z], dim=1)
        else:
            zeros = torch.zeros_like(input)
            mean, log_sd = self.prior(zeros).chunk(2, 1)
            z = self.gaussian_sample(z_in, mean, log_sd)
            input = z
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        b, c, h, w = input.shape 
        inp_unsq = input.view(b, c // 4, 2, 2, h, w)
        inp_unsq = inp_unsq.permute(0, 1, 4, 2, 5, 3)
        inp_unsq = inp_unsq.contiguous().view(b, c // 4, h * 2, w * 2)
        return inp_unsq 

    def gaussian_log_p(self, z, mean, log_sd):
        return -0.5 * log(2 * pi) - log_sd - 0.5 * (z - mean) ** 2 / torch.exp(2 * log_sd) 

    def gaussian_sample(self, z, mean, log_sd):
        return mean + torch.exp(log_sd) * z
  

class Network(nn.Module):
    def __init__(self, num_blocks=4, num_flows=32, inf=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_flows = num_flows 
        self.inf = inf 
        self.blocks = nn.ModuleList()
        ch = 3 
        for i in range(self.num_blocks - 1):
            self.blocks.append(Block(num_flows=self.num_flows,
                                    ch=ch, inf=inf))
            ch *= 2 
        self.blocks.append(Block(num_flows=self.num_flows,
                                ch=ch, split=False, inf=self.inf))

    def forward(self, x):
        log_p_sum = 0 
        logdet = 0
        out = x 
        z_outs = []
        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet += det 
            log_p_sum += log_p 
        return log_p_sum, logdet, z_outs 

    def reverse(self, z_list):
        input = z_list[-1]
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_in=input, input=input)
            else:
                input = block.reverse(z_in=z_list[-i-1], input=input)
        return input 
            

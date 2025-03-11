import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Helper function to initialize convolutional layers
def _init_conv_layer(conv, activation, mode='fan_out'):
    """ Initialize convolutional weights appropriately. """
    if isinstance(activation, nn.LeakyReLU):
        nn.init.kaiming_uniform_(conv.weight, a=activation.negative_slope, nonlinearity='leaky_relu', mode=mode)
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu', mode=mode)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)

# Convert output tensor to image format
def output_to_image(out):
    """ Convert torch tensor output to image format. """
    return ((out[0].cpu().permute(1, 2, 0) + 1.) * 127.5).to(torch.uint8).numpy()

# Define a generic Gated Convolutional Layer
class GConv(nn.Module):
    """ Implements the gated 2D convolution. """
    def __init__(self, cnum_in, cnum_out, ksize, stride=1, padding='auto', rate=1, activation=nn.ELU(), bias=True):
        super().__init__()
        padding = rate * (ksize - 1) // 2 if padding == 'auto' else padding
        self.activation = activation
        self.cnum_out = cnum_out
        num_conv_out = cnum_out if cnum_out == 3 or activation is None else 2 * cnum_out
        self.conv = nn.Conv2d(cnum_in, num_conv_out, kernel_size=ksize, stride=stride, padding=padding, dilation=rate, bias=bias)
        _init_conv_layer(self.conv, activation)

    def forward(self, x):
        x = self.conv(x)
        if self.cnum_out == 3 or self.activation is None:
            return x
        x, y = torch.split(x, self.cnum_out, dim=1)
        return self.activation(x) * torch.sigmoid(y)

# Define Gated Deconvolution for upsampling
class GDeConv(nn.Module):
    """ Upsampling followed by a gated convolution. """
    def __init__(self, cnum_in, cnum_out, padding=1):
        super().__init__()
        self.conv = GConv(cnum_in, cnum_out, 3, 1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

# Define Downsampling Block using gated convolutions
class GDownsamplingBlock(nn.Module):
    """ Implements a block of operations for downsampling. """
    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden is None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, 3, 2)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, 1)

    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x

# Define Upsampling Block using gated convolutions
class GUpsamplingBlock(nn.Module):
    """ Implements a block of operations for upsampling. """
    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden is None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, 1)

    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        return x

# Define Coarse Generator using sequential blocks
class CoarseGenerator(nn.Module):
    """ Coarse generator for structured inpainting. """
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = GConv(cnum_in, cnum//2, 5, 1, padding=2)
        self.down_block1 = GDownsamplingBlock(cnum//2, cnum)
        self.down_block2 = GDownsamplingBlock(cnum, 2 * cnum)
        # Bottleneck layers
        self.conv_bn1 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv_bn2 = GConv(2 * cnum, 2 * cnum, 3, rate=2, padding=2)
        self.conv_bn3 = GConv(2 * cnum, 2 * cnum, 3, rate=4, padding=4)
        self.conv_bn4 = GConv(2 * cnum, 2 * cnum, 3, rate=8, padding=8)
        self.conv_bn5 = GConv(2 * cnum, 2 * cnum, 3, rate=16, padding=16)
        self.conv_bn6 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv_bn7 = GConv(2 * cnum, 2 * cnum, 3, 1)
        # Upsampling layers
        self.up_block1 = GUpsamplingBlock(2 * cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)
        # Final RGB conversion
        self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = self.conv_bn5(x)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)
        x = self.up_block1(x)
        x = self.up_block2(x)
        x = self.conv_to_rgb(x)
        return self.tanh(x)

# Define Fine Generator with additional contextual attention
class FineGenerator(nn.Module):
    """ Fine generator with contextual attention. """
    def __init__(self, cnum, return_flow=False):
        super().__init__()
        self.return_flow = return_flow
        # Convolutional branch
        self.conv_conv1 = GConv(3, cnum//2, 5, 1, padding=2)
        self.conv_down_block1 = GDownsamplingBlock(cnum//2, cnum, cnum_hidden=cnum//2)
        self.conv_down_block2 = GDownsamplingBlock(cnum, 2 * cnum, cnum_hidden=cnum)
        self.conv_conv_bn1 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv_conv_bn2 = GConv(2 * cnum, 2 * cnum, 3, rate=2, padding=2)
        self.conv_conv_bn3 = GConv(2 * cnum, 2 * cnum, 3, rate=4, padding=4)
        self.conv_conv_bn4 = GConv(2 * cnum, 2 * cnum, 3, rate=8, padding=8)
        self.conv_conv_bn5 = GConv(2 * cnum, 2 * cnum, 3, rate=16, padding=16)
        
        # Attention branch
        self.ca_conv1 = GConv(3, cnum//2, 5, 1, padding=2)
        self.ca_down_block1 = GDownsamplingBlock(cnum//2, cnum, cnum_hidden=cnum//2)
        self.ca_down_block2 = GDownsamplingBlock(cnum, 2 * cnum)
        self.ca_conv_bn1 = GConv(2 * cnum, 2 * cnum, 3, 1, activation=nn.ReLU())
        self.contextual_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True, return_flow=return_flow, n_down=2)
        self.ca_conv_bn4 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.ca_conv_bn5 = GConv(2 * cnum, 2 * cnum, 3, 1)

        # Combined branches
        self.conv_bn6 = GConv(4 * cnum, 2 * cnum, 3, 1)
        self.conv_bn7 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.up_block1 = GUpsamplingBlock(2 * cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)
        self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        x_conv = self.conv_conv1(x)
        # Downsampling
        x_conv = self.conv_down_block1(x_conv)
        x_conv = self.conv_down_block2(x_conv)
        # Bottleneck
        x_conv = self.conv_conv_bn1(x_conv)
        x_conv = self.conv_conv_bn2(x_conv)
        x_conv = self.conv_conv_bn3(x_conv)
        x_conv = self.conv_conv_bn4(x_conv)
        x_conv = self.conv_conv_bn5(x_conv)

        # Attention
        x_ca = self.ca_conv1(x)
        x_ca = self.ca_down_block1(x_ca)
        x_ca = self.ca_down_block2(x_ca)
        x_ca = self.ca_conv_bn1(x_ca)
        x_ca, offset_flow = self.contextual_attention(x_ca, x_ca, mask)
        x_ca = self.ca_conv_bn4(x_ca)
        x_ca = self.ca_conv_bn5(x_ca)
        
        # Combine branches
        x_combined = torch.cat([x_conv, x_ca], dim=1)
        x_combined = self.conv_bn6(x_combined)
        x_combined = self.conv_bn7(x_combined)
        # Upsampling
        x_combined = self.up_block1(x_combined)
        x_combined = self.up_block2(x_combined)
        # Convert to RGB
        x_combined = self.conv_to_rgb(x_combined)
        return self.tanh(x_combined), offset_flow if self.return_flow else None

# Define the main generator class
class Generator(nn.Module):
    """ The main generator class using a coarse and fine generator for inpainting. """
    def __init__(self, cnum_in=5, cnum=48, return_flow=False, checkpoint=None):
        super().__init__()
        self.stage1 = CoarseGenerator(cnum_in, cnum)
        self.stage2 = FineGenerator(cnum, return_flow)
        self.return_flow = return_flow
        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint, map_location='cpu')['G']
            self.load_state_dict(generator_state_dict, strict=True)
        self.eval()

    def forward(self, x, mask):
        xin = x
        x_stage1 = self.stage1(x)
        x_inpainted = x_stage1 * mask + xin[:, :3, :, :] * (1 - mask)
        x_stage2, offset_flow = self.stage2(x_inpainted, mask)
        if self.return_flow:
            return x_stage1, x_stage2, offset_flow
        return x_stage1, x_stage2

    @torch.inference_mode()
    def infer(self, image, mask, return_vals=['inpainted', 'stage1'], device='cuda'):
        _, h, w = image.shape
        grid = 8
        image = image[:3, :h // grid * grid, :w // grid * grid].unsqueeze(0)
        mask = mask[:1, :h // grid * grid, :w // grid * grid].unsqueeze(0)
        image = image * 2 - 1
        mask = (mask > 0).float()
        image_masked = image * (1 - mask)
        ones_x = torch.ones_like(image_masked)[:, :1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)
        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)
        image_compl = image * (1 - mask) + x_stage2 * mask
        output = []
        for return_val in return_vals:
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')
        return output

# Contextual Attention implementation
class ContextualAttention(nn.Module):
    """ Contextual attention layer implementation for inpainting. """
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., n_down=2, fuse=False, return_flow=False, device_ids=None):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.n_down = n_down
        self.return_flow = return_flow
        self.register_buffer('fuse_weight', torch.eye(fuse_k).view(1, 1, fuse_k, fuse_k))

    def forward(self, f, b, mask=None):
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())
        kernel = 2 * self.rate
        raw_w = extract_image_patches(b, ksize=kernel, stride=self.rate * self.stride, rate=1, padding='auto')
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1).permute(0, 4, 1, 2, 3)
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')
        int_fs, int_bs = list(f.size()), list(b.size())
        w = extract_image_patches(b, ksize=self.ksize, stride=self.stride, rate=1, padding='auto')
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1).permute(0, 4, 1, 2, 3)
        mask = F.interpolate(mask, scale_factor=1. / ((2**self.n_down) * self.rate), mode='nearest') if mask is not None else torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]], f.device)

        scale = self.softmax_scale
        y, offsets = [], []
        mm = (torch.mean((mask.unsqueeze(2) > 0).float(), dim=[1, 2, 3], keepdim=True) == 0.).to(torch.float32).permute(1, 0, 2, 3)
        for xi, wi, raw_wi in zip(f.split(1, dim=0), w.split(1, dim=0), raw_w.split(1, dim=0)):
            wi = wi[0]
            wi_normed = wi / wi.norm(p=2, dim=[1, 2, 3], keepdim=True).clamp_min(1e-4)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=(self.ksize-1)//2)
            if self.fuse:
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = F.conv2d(yi, self.fuse_weight, stride=1, padding=(self.fuse_k-1)//2)
                yi = yi.permute(0, 2, 1, 4, 3).contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = F.conv2d(yi, self.fuse_weight, stride=1, padding=(self.fuse_k-1)//2).view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = (yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3]) * mm).softmax(dim=1) * mm
            if self.return_flow:
                offset = yi.argmax(dim=1, keepdim=True)
                times = (int_fs[2] * int_fs[3]) // (int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'), offset % int_fs[3]], dim=1)
                offsets.append(offset)
            y.append(F.conv_transpose2d(yi, raw_wi[0], stride=self.rate, padding=1) / 4.)

        flow = self._compute_offsets_flow(offsets, int_fs, raw_int_fs, f.device) if self.return_flow else None
        return torch.cat(y, dim=0).view(raw_int_fs), flow

    def _compute_offsets_flow(self, offsets, int_fs, raw_int_fs, device):
        offsets = torch.cat(offsets, dim=0).view(int_fs[0], 2, *int_fs[2:])
        h_add = torch.arange(int_fs[2], device=device).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        flow = self._to_flow_image(offsets.permute(0, 2, 3, 1).cpu().numpy())
        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate, mode='bilinear', align_corners=True)
        return flow

    def _to_flow_image(self, flow):
        # Conversion to flow image, placeholder for actual visualization code.
        return torch.from_numpy(flow).permute(0, 3, 1, 2)
        

def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img

# ----------------------------------------------------------------------------

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

# ----------------------------------------------------------------------------


def extract_image_patches(images, ksize, stride, rate, padding='auto'):
    padding = rate*(ksize-1)//2 if padding == 'auto' else padding

    unfold = torch.nn.Unfold(kernel_size=ksize,
                             dilation=rate,
                             padding=padding,
                             stride=stride)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

# ----------------------------------------------------------------------------

#################################
######### DISCRIMINATOR #########
#################################

class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in,
                 cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        weight_orig = self.weight.flatten(1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x

# ----------------------------------------------------------------------------

class DConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out, ksize=5, stride=2, padding='auto'):
        super().__init__()
        padding = (ksize-1)//2 if padding == 'auto' else padding
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)
        #self.conv_sn = spectral_norm(nn.Conv2d(cnum_in, cnum_out, ksize, stride, padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x

# ----------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)
        self.conv6 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.Flatten()(x)

        return x

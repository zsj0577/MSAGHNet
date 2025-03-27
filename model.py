import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


nonlinearity = partial(F.relu, inplace=True)

#### Transformer
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MCT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=512, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.out = Rearrange("b (h w) c->b c h w", h=image_height // patch_height, w=image_width // patch_width)

        # 这里上采样倍数为8倍。为了保持和图中的feature size一样
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=patch_size // 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU())

    def forward(self, img):
        # 这里对应了图中的Linear Projection，主要是将图片分块嵌入，成为一个序列
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # 为图像切片序列加上索引
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # 输入到Transformer中处理
        x = self.transformer(x)

        # delete cls_tokens, 输出前需要删除掉索引
        output = x[:, 1:, :]
        # print(x.shape, output.shape)
        output = self.out(output)

        # Transformer输出后，上采样到原始尺寸
        output = self.upsample(output)
        output = self.conv(output)

        return output

class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class CAMLayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        channel_x = channel_out * x

        return channel_x

class MSFA(nn.Module):
    def __init__(self, ch_inannel=512, out_channel=512):
        super(MSFA, self).__init__()

        depth = out_channel // 4
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = nn.Conv2d(ch_inannel, depth, 1, 1)

        self.atrous_block1 = nn.Conv2d(ch_inannel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(ch_inannel, depth, 3, 1, padding=1, dilation=1)
        self.atrous_block12 = nn.Conv2d(ch_inannel, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block18 = nn.Conv2d(ch_inannel, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block21 = nn.Conv2d(ch_inannel, depth, 3, 1, padding=8, dilation=8)
        self.attention = nn.Conv2d(depth * 4, 4, 1, padding=0, groups=4, bias=False)
        self.conv_1x1_output = nn.Conv2d(depth * 6, out_channel, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]

        p1 = self.pool(x)
        p1 = self.conv(p1)
        # image_features = F.interpolate(image_features, size=size, mode='bilinear')

        d0 = self.atrous_block1(x)
        d1 = self.atrous_block6(x)
        d2 = self.atrous_block12(x)
        d3 = self.atrous_block18(x)
        d4 = self.atrous_block21(x)
        # print(d0.shape, d1.shape, d2.shape, d3.shape, d4.shape, p1.shape,)
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)

        net = self.conv_1x1_output(torch.cat([d0, d1, d2, d3, d4, p1], dim=1))
        out = self.relu(net)

        return out

class GCFF_block(nn.Module):
    def __init__(self, ch_in1, ch_in2, ch_out):
        super(GCFF_block, self).__init__()

        self.W_e = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out)
        )
        self.W_d = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in2, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out)
        )
        self.relu = nn.ReLU(inplace=True)
        self.gc = ContextBlock(inplanes=ch_out, ratio=1. / 8., pooling_type='att')
        self.ca = CAMLayer(ch_out)

        self.psi = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, e, d):
        e1 = self.W_e(e)
        d1 = self.W_d(d)
        x1 = self.relu(e1 + d1)
        out = self.psi(self.gc(x1) + self.ca(x1)) + e1

        return out

def Tensor_Resize(input):
    scaled_2 = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
    scaled_4 = F.interpolate(input, scale_factor=0.25, mode='bilinear', align_corners=False)
    scaled_8 = F.interpolate(input, scale_factor=0.125, mode='bilinear', align_corners=False)
    scaled_16 = F.interpolate(input, scale_factor=0.0625, mode='bilinear', align_corners=False)

    return scaled_2, scaled_4, scaled_8, scaled_16

class ChannelShuffle(nn.Module):
    def __init__(self, groups=4):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        # 分组参数
        batch_size, num_channels, height, width = x.size()
        num_channels_per_group = num_channels // self.groups
        # 重塑输入
        x = x.view(batch_size, self.groups, num_channels_per_group, height, width)
        # 转置并重塑
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

class SE_MSFE(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SE_MSFE, self).__init__()

        self.W_e = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        # self.relu = nn.ReLU()
        self.con1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.con3 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        self.con5 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        self.con7 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        self.con9 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = ChannelShuffle(4)
        self.se = SEModule(ch_out)


    def forward(self, x1):
        x1 = self.W_e(x1)
        c1 = self.con1(x1)
        c3 = self.con3(x1)
        c5 = self.con5(x1)
        c7 = self.con7(x1)
        c9 = self.con9(x1)
        out = self.se(self.shuffle(torch.cat([c3, c5, c7, c9], dim=1)) + c1)

        return out

class DPDS_Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(DPDS_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=2, padding=1, groups=out_planes, bias=False)
        self.conv0 = nn.Conv2d(out_planes, out_planes, 1, stride=1, padding=0, groups=1, bias=False)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1(self.conv0(input))
        output = self.act(self.bn(output + self.Maxpool(input)))
        return output

class OutConv(nn.Module):
    def __init__(self, ch_in, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, out_ch, 3, padding=1),
        )

    def forward(self, input):
        return self.conv(input)


class Conv3x3_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv3x3_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU(),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
    def forward(self, input):

        return self.conv(input)


class Decoder_Block_IN(nn.Module):
    def __init__(self, num_channels=3, num_classes=1):
        super(Decoder_Block_IN, self).__init__()

        filters = [16, 32, 64, 128, 256, 512]

        self.msa4 = GCFF_block(filters[4], filters[4], filters[4])
        self.msa3 = GCFF_block(filters[3], filters[4], filters[3])
        self.msa2 = GCFF_block(filters[2], filters[3], filters[2])
        self.msa1 = GCFF_block(filters[1], filters[2], filters[1])
        self.msa0 = GCFF_block(filters[0], filters[1], filters[0])

        self.finalconv1 = nn.Conv2d(filters[0], filters[0], 3, 1, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0], num_classes, 3, padding=1)

        self.out4 = OutConv(filters[4], 1)
        self.out3 = OutConv(filters[3], 1)
        self.out2 = OutConv(filters[2], 1)
        self.out1 = OutConv(filters[1], 1)

    def forward(self, x, e1, e2, e3, e4, e5):
        d4 = self.msa4(e4, e5)
        # d4 = self.res4(d4)
        d3 = self.msa3(e3, d4)
        # d3 = self.res3(d3)
        d2 = self.msa2(e2, d3)
        # d2 = self.res2(d2)
        d1 = self.msa1(e1, d2)

        d0 = self.msa0(x, d1)

        out = self.finalconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)

        # print(e5.shape, d4.shape, d3.shape, d2.shape, d1.shape, out.shape)
        # exit()
        out4 = self.out4(d4)
        out3 = self.out3(d3)
        out2 = self.out2(d2)
        out1 = self.out1(d1)

        return out, out1, out2, out3, out4
        # return out


class MSAGHNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1):
        super(MSAGHNet, self).__init__()

        filters = [16, 32, 64, 128, 256, 512]
        self.firstconv1 = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU()
        )
        self.conv_x2 = DoubleConv(3, filters[0])
        self.conv_x4 = DoubleConv(3, filters[1])
        self.conv_x8 = DoubleConv(3, filters[2])
        self.conv_x16 = DoubleConv(3, filters[3])

        self.encoder1 = SE_MSFE(filters[0] * 2, filters[1])
        self.encoder2 = SE_MSFE(filters[1] * 2, filters[2])
        self.encoder3 = SE_MSFE(filters[2] * 2, filters[3])
        self.encoder4 = SE_MSFE(filters[3] * 2, filters[4])

        self.down1 = DPDS_Block(filters[0], filters[0])
        self.down2 = DPDS_Block(filters[1], filters[1])
        self.down3 = DPDS_Block(filters[2], filters[2])
        self.down4 = DPDS_Block(filters[3], filters[3])
        self.down5 = DPDS_Block(filters[4], filters[4])

        self.dblock = MSFA(filters[4], filters[4])
        self.vit = MCT(image_size=(16, 16), patch_size=8, channels=filters[4], dim=filters[4], depth=12, heads=16,
                       mlp_dim=1024,
                       dropout=0.1, emb_dropout=0.1)

        self.catconv = nn.Sequential(
            nn.Conv2d(filters[4], filters[4], kernel_size=1, stride=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )

        self.shuffle = ChannelShuffle(4)

        self.decoder1 = Decoder_Block_IN()

    def forward(self, x):
        x_2, x_4, x_8, x_16 = Tensor_Resize(x)

        x = self.firstconv1(x)

        d1 = torch.cat([self.conv_x2(x_2), self.down1(x)], dim=1)
        # d1 = self.conv_x2(x_2) + self.down1(x)
        e1 = self.encoder1(d1)

        d2 = torch.cat([self.conv_x4(x_4), self.down2(e1)], dim=1)
        # d2 = self.conv_x4(x_4) + self.down2(e1)
        e2 = self.encoder2(d2)

        d3 = torch.cat([self.conv_x8(x_8), self.down3(e2)], dim=1)
        # d3 = self.conv_x8(x_8) + self.down3(e2)
        e3 = self.encoder3(d3)

        d4 = torch.cat([self.conv_x16(x_16), self.down4(e3)], dim=1)
        # d4 = self.conv_x16(x_16) + self.down4(e3)
        e4 = self.encoder4(d4)

        d5 = self.down5(e4)
        m5 = self.dblock(d5)
        v5 = self.vit(d5)

        e5 = torch.relu(m5 + v5)

        out, out1, out2, out3, out4 = self.decoder1(x, e1, e2, e3, e4, e5)

        return out, out1, out2, out3, out4


if __name__ == '__main__':
    x = torch.rand([1, 3, 512, 512])
    model = MSAGHNet(3, 1)
    pred1, pred2, _, _, _ = model(x)
    print(pred1.shape, pred2.shape)



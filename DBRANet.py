import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu,inplace=True)
#import thop

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))

        return x

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        e1 = self.encoder1(x) #128*128*64
        e2 = self.encoder2(e1) #64*64*128
        e3 = self.encoder3(e2) #32*32*256
        e4 = self.encoder4(e3) #16*16*512

        # Decoder
        d4 = self.decoder4(e4) + e3 #32*32*256
        d3 = self.decoder3(d4) + e2 #64*64*128
        d2 = self.decoder2(d3) + e1 #128*128*64
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class RefinedAsymmetricBlock:
    def __init__(self):
        self.inplanes = 64

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class BasicAsymmetricBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False):
        super(BasicAsymmetricBlock, self).__init__()
        dim_out = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3),
                               stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, 3),
                                 stride=1, padding=(0, 1), bias=bias)
        self.conv2_2 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(3, 1),
                                 stride=1, padding=(1, 0), bias=bias)
        self.conv2_3 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(3, 1),
                                 stride=1, padding=(1, 0), bias=bias)
        self.conv2_4 = nn.Conv2d(dim_out, dim_out // 4, kernel_size=(1, 3),
                                 stride=1, padding=(0, 1), bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x0 = self.conv2(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(self.h_transform(x))
        x3 = self.inv_h_transform(x3)
        x4 = self.conv2_4(self.v_transform(x))
        x4 = self.inv_v_transform(x4)

        x = x0 + torch.cat((x1, x2, x3, x4), 1)
        out = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, pool_size):
        super(AttentionDecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        self.lad = LocalAttentionDecoder(n_filters,pool_size)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.lad(x)

        return x

class LocalAttentionDecoder(nn.Module):
    def __init__(self, in_dim, pool_size):
        super(LocalAttentionDecoder, self).__init__()
        self.conv1x1 = nn.Conv2d(in_dim, in_dim, 1)
        self.localAvgPool = nn.AdaptiveAvgPool2d(pool_size)
        self.localMaxPool = nn.AdaptiveMaxPool2d(pool_size)
        # self.conv2 = nn.Conv2d(in_dim, in_dim, 1)
    def _generate_sequence(self, pooling_map_size, basic):
        sq = []
        for i in range(pooling_map_size):
            for j in range(basic):
                sq.append(pooling_map_size*j+i)
        return sq

    def _enlarge_map(self, pooling_map, feature_map_size):

        B,C,H,_= pooling_map.size()
        basic = int(feature_map_size//H)
        a = pooling_map.view(B,C,1,-1).repeat(1,1,basic,1).view(B,C,-1,1).repeat(1,1,1,basic).view(B,C,feature_map_size,feature_map_size)
        b = torch.tensor(self._generate_sequence(H,basic))
        a = torch.index_select(a, dim=2, index=b.cuda())
        # a = torch.index_select(a, dim=2, index=b)
        return a
        
    def forward(self, x):
        st = x
        H = x.size(2)
        x_lap = self.localAvgPool(x)
        x_lmp = self.localMaxPool(x)
        with torch.no_grad():
            x = self._enlarge_map(x_lap + x_lmp, H)
            #x = self._enlarge_map(self.conv2(x_lap + x_lmp), H)
        x = torch.mul(st,torch.sigmoid(x))

        return  x

class DBRANet_dual(nn.Module):
    def __init__(self, num_classes=1):
        super(DBRANet_dual, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        layers = [3, 4, 6, 3]
        top_residual_branch = models.resnet34(pretrained=True)
        bottom_refinedasymmetric_branch = RefinedAsymmetricBlock()

        #公共部分，resnet34 stage1
        self.firstconv = top_residual_branch.conv1
        self.firstbn = top_residual_branch.bn1
        self.firstrelu = top_residual_branch.relu
        self.firstmaxpool = top_residual_branch.maxpool

        #Top residual branch
        self.residual1 = top_residual_branch.layer1
        self.residual2 = top_residual_branch.layer2
        self.residual3 = top_residual_branch.layer3
        self.residual4 = top_residual_branch.layer4

        #Bottom refinedasymmetric branch
        self.asymmetric1 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 64, layers[0])
        self.asymmetric2 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 128, layers[1], stride=2)
        self.asymmetric3 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 256, layers[2], stride=2)
        self.asymmetric4 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 512, layers[3], stride=2)

        self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        
        #Top
        top_e1 = self.residual1(x) #128*128*64
        top_e2 = self.residual2(top_e1) #64*64*128
        top_e3 = self.residual3(top_e2) #32*32*256
        top_e4 = self.residual4(top_e3) #16*16*512

        #Bottom
        bottom_e1 = self.asymmetric1(x) #128*128*64
        bottom_e2 = self.asymmetric2(bottom_e1) #64*64*128
        bottom_e3 = self.asymmetric3(bottom_e2) #32*32*256
        bottom_e4 = self.asymmetric4(bottom_e3) #16*16*512

                #Fusion
        fusion1 = torch.cat((top_e1,bottom_e1),1) #128*128*128
        fusion2 = torch.cat((top_e2,bottom_e2),1) #64*64*256
        fusion3 = torch.cat((top_e3,bottom_e3),1) #32*32*512
        fusion4 = torch.cat((top_e4,bottom_e4),1) #16*16*1024

        # Decoder
        d4 = self.decoder4(fusion4) + fusion3 #32*32*512
        d3 = self.decoder3(d4) + fusion2 #64*64*256
        d2 = self.decoder2(d3) + fusion1 #128*128*128
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class DBRANet_dual_PW(nn.Module):
    def __init__(self, num_classes=1):
        super(DBRANet_dual_PW, self).__init__()

        filters = [64, 128, 256, 512]
        layers = [3, 4, 6, 3]
        top_residual_branch = models.resnet34(pretrained=True)
        bottom_refinedasymmetric_branch = RefinedAsymmetricBlock()

        #公共部分，resnet34 stage1
        self.firstconv = top_residual_branch.conv1
        self.firstbn = top_residual_branch.bn1
        self.firstrelu = top_residual_branch.relu
        self.firstmaxpool = top_residual_branch.maxpool

        #Top residual branch
        self.residual1 = top_residual_branch.layer1
        self.residual2 = top_residual_branch.layer2
        self.residual3 = top_residual_branch.layer3
        self.residual4 = top_residual_branch.layer4

        #Bottom refinedasymmetric branch
        self.asymmetric1 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 64, layers[0])
        self.asymmetric2 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 128, layers[1], stride=2)
        self.asymmetric3 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 256, layers[2], stride=2)
        self.asymmetric4 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 512, layers[3], stride=2)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        
        #Top
        top_e1 = self.residual1(x) #128*128*64
        top_e2 = self.residual2(top_e1) #64*64*128
        top_e3 = self.residual3(top_e2) #32*32*256
        top_e4 = self.residual4(top_e3) #16*16*512

        #Bottom
        bottom_e1 = self.asymmetric1(x) #128*128*64
        bottom_e2 = self.asymmetric2(bottom_e1) #64*64*128
        bottom_e3 = self.asymmetric3(bottom_e2) #32*32*256
        bottom_e4 = self.asymmetric4(bottom_e3) #16*16*512

                #Fusion
        #Fusion
        fusion1 = top_e1 + bottom_e1 #128*128*128
        fusion2 = top_e2 + bottom_e2 #64*64*256
        fusion3 = top_e3 + bottom_e3 #32*32*512
        fusion4 = top_e4 + bottom_e4 #16*16*1024

        # Decoder
        d4 = self.decoder4(fusion4) + fusion3 #32*32*512
        d3 = self.decoder3(d4) + fusion2 #64*64*256
        d2 = self.decoder2(d3) + fusion1 #128*128*128
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class RACBranch(nn.Module):
    def __init__(self, num_classes=1):
        super(RACBranch, self).__init__()

        filters = [64, 128, 256, 512]
        layers = [3, 4, 6, 3]
        top_residual_branch = models.resnet34(pretrained=True)
        bottom_refinedasymmetric_branch = RefinedAsymmetricBlock()

        #公共部分，resnet34 stage1
        self.firstconv = top_residual_branch.conv1
        self.firstbn = top_residual_branch.bn1
        self.firstrelu = top_residual_branch.relu
        self.firstmaxpool = top_residual_branch.maxpool

        #Bottom refinedasymmetric branch
        self.asymmetric1 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 64, layers[0])
        self.asymmetric2 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 128, layers[1], stride=2)
        self.asymmetric3 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 256, layers[2], stride=2)
        self.asymmetric4 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 512, layers[3], stride=2)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        #Bottom
        bottom_e1 = self.asymmetric1(x) #128*128*64
        bottom_e2 = self.asymmetric2(bottom_e1) #64*64*128
        bottom_e3 = self.asymmetric3(bottom_e2) #32*32*256
        bottom_e4 = self.asymmetric4(bottom_e3) #16*16*512

        # Decoder
        d4 = self.decoder4(bottom_e4) + bottom_e3 #32*32*512
        d3 = self.decoder3(d4) + bottom_e2 #64*64*256
        d2 = self.decoder2(d3) + bottom_e1 #128*128*128
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class DBRANet(nn.Module):
    def __init__(self, pool_size):
        super(DBRANet, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        layers = [3, 4, 6, 3]
        top_residual_branch = models.resnet34(pretrained=False)
        top_residual_branch.load_state_dict(torch.load('./networks/resnet34.pth'))
        bottom_refinedasymmetric_branch = RefinedAsymmetricBlock()

        #公共部分，resnet34 stage1
        self.firstconv = top_residual_branch.conv1
        self.firstbn = top_residual_branch.bn1
        self.firstrelu = top_residual_branch.relu
        self.firstmaxpool = top_residual_branch.maxpool

        #Top residual branch
        self.residual1 = top_residual_branch.layer1
        self.residual2 = top_residual_branch.layer2
        self.residual3 = top_residual_branch.layer3
        self.residual4 = top_residual_branch.layer4

        #Bottom refinedasymmetric branch
        self.asymmetric1 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 64, layers[0])
        self.asymmetric2 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 128, layers[1], stride=2)
        self.asymmetric3 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 256, layers[2], stride=2)
        self.asymmetric4 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 512, layers[3], stride=2)

        self.decoder4 = AttentionDecoderBlock(filters[4], filters[3], pool_size)
        self.decoder3 = AttentionDecoderBlock(filters[3], filters[2], pool_size)
        self.decoder2 = AttentionDecoderBlock(filters[2], filters[1], pool_size)
        self.decoder1 = AttentionDecoderBlock(filters[1], filters[0], pool_size)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        
        #Top
        top_e1 = self.residual1(x) #128*128*64
        top_e2 = self.residual2(top_e1) #64*64*128
        top_e3 = self.residual3(top_e2) #32*32*256
        top_e4 = self.residual4(top_e3) #16*16*512

        #Bottom
        bottom_e1 = self.asymmetric1(x) #128*128*64
        bottom_e2 = self.asymmetric2(bottom_e1) #64*64*128
        bottom_e3 = self.asymmetric3(bottom_e2) #32*32*256
        bottom_e4 = self.asymmetric4(bottom_e3) #16*16*512

        #Fusion
        fusion1 = torch.cat((top_e1,bottom_e1),1) #128*128*128
        fusion2 = torch.cat((top_e2,bottom_e2),1) #64*64*256
        fusion3 = torch.cat((top_e3,bottom_e3),1) #32*32*512
        fusion4 = torch.cat((top_e4,bottom_e4),1) #16*16*1024

        # Decoder
        d4 = self.decoder4(fusion4) + fusion3 #32*32*512
        d3 = self.decoder3(d4) + fusion2 #64*64*256
        d2 = self.decoder2(d3) + fusion1 #128*128*128
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class DBRANet_PW(nn.Module):
    def __init__(self, pool_size):
        super(DBRANet_PW, self).__init__()

        filters = [64, 128, 256, 512]
        layers = [3, 4, 6, 3]
        top_residual_branch = models.resnet34(pretrained=True)
        bottom_refinedasymmetric_branch = RefinedAsymmetricBlock()

        #公共部分，resnet34 stage1
        self.firstconv = top_residual_branch.conv1
        self.firstbn = top_residual_branch.bn1
        self.firstrelu = top_residual_branch.relu
        self.firstmaxpool = top_residual_branch.maxpool

        #Top residual branch
        self.residual1 = top_residual_branch.layer1
        self.residual2 = top_residual_branch.layer2
        self.residual3 = top_residual_branch.layer3
        self.residual4 = top_residual_branch.layer4

        #Bottom refinedasymmetric branch
        self.asymmetric1 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 64, layers[0])
        self.asymmetric2 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 128, layers[1], stride=2)
        self.asymmetric3 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 256, layers[2], stride=2)
        self.asymmetric4 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 512, layers[3], stride=2)

        self.decoder4 = AttentionDecoderBlock(filters[3], filters[2], pool_size)
        self.decoder3 = AttentionDecoderBlock(filters[2], filters[1], pool_size)
        self.decoder2 = AttentionDecoderBlock(filters[1], filters[0], pool_size)
        self.decoder1 = AttentionDecoderBlock(filters[0], filters[0], pool_size)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        
        #Top
        top_e1 = self.residual1(x) #128*128*64
        top_e2 = self.residual2(top_e1) #64*64*128
        top_e3 = self.residual3(top_e2) #32*32*256
        top_e4 = self.residual4(top_e3) #16*16*512

        #Bottom
        bottom_e1 = self.asymmetric1(x) #128*128*64
        bottom_e2 = self.asymmetric2(bottom_e1) #64*64*128
        bottom_e3 = self.asymmetric3(bottom_e2) #32*32*256
        bottom_e4 = self.asymmetric4(bottom_e3) #16*16*512

        #Fusion
        fusion1 = top_e1 + bottom_e1 #128*128*128
        fusion2 = top_e2 + bottom_e2 #64*64*256
        fusion3 = top_e3 + bottom_e3 #32*32*512
        fusion4 = top_e4 + bottom_e4 #16*16*1024

        # Decoder
        d4 = self.decoder4(fusion4) + fusion3 #32*32*512
        d3 = self.decoder3(d4) + fusion2 #64*64*256
        d2 = self.decoder2(d3) + fusion1 #128*128*128
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class DBRANet_RAD(nn.Module):
    def __init__(self, pool_size):
        super(DBRANet_RAD, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = AttentionDecoderBlock(filters[3], filters[2], pool_size)
        self.decoder3 = AttentionDecoderBlock(filters[2], filters[1], pool_size)
        self.decoder2 = AttentionDecoderBlock(filters[1], filters[0], pool_size)
        self.decoder1 = AttentionDecoderBlock(filters[0], filters[0], pool_size)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        e1 = self.encoder1(x) #128*128*64
        e2 = self.encoder2(e1) #64*64*128
        e3 = self.encoder3(e2) #32*32*256
        e4 = self.encoder4(e3) #16*16*512

        # Decoder
        d4 = self.decoder4(e4) + e3 #32*32*256
        d3 = self.decoder3(d4) + e2 #64*64*128
        d2 = self.decoder2(d3) + e1 #128*128*64
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

class DBRANet_PW_Fuse(nn.Module):
    def __init__(self, pool_size=4):
        super(DBRANet_PW_Fuse, self).__init__()

        filters = [64, 128, 256, 512]
        layers = [3, 4, 6, 3]
        top_residual_branch = models.resnet34(pretrained=True)
        bottom_refinedasymmetric_branch = RefinedAsymmetricBlock()

        #公共部分，resnet34 stage1
        self.firstconv = top_residual_branch.conv1
        self.firstbn = top_residual_branch.bn1
        self.firstrelu = top_residual_branch.relu
        self.firstmaxpool = top_residual_branch.maxpool

        #Top residual branch
        self.residual1 = top_residual_branch.layer1
        self.residual2 = top_residual_branch.layer2
        self.residual3 = top_residual_branch.layer3
        self.residual4 = top_residual_branch.layer4

        #Bottom refinedasymmetric branch
        self.asymmetric1 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 64, layers[0])
        self.asymmetric2 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 128, layers[1], stride=2)
        self.asymmetric3 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 256, layers[2], stride=2)
        self.asymmetric4 = bottom_refinedasymmetric_branch._make_layer(BasicAsymmetricBlock, 512, layers[3], stride=2)

        self.decoder4 = AttentionDecoderBlock(filters[3], filters[2], pool_size)
        self.decoder3 = AttentionDecoderBlock(filters[2], filters[1], pool_size)
        self.decoder2 = AttentionDecoderBlock(filters[1], filters[0], pool_size)
        self.decoder1 = AttentionDecoderBlock(filters[0], filters[0], pool_size)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #256*256*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) #128*128*64
        
        #Top #Bottom
        top_e1 = self.residual1(x) #128*128*64
        bottom_e1 = self.asymmetric1(x) #128*128*64
        e1 = top_e1 + bottom_e1

        top_e2 = self.residual2(e1) #64*64*128
        bottom_e2 = self.asymmetric2(e1) #64*64*128
        e2 = top_e2 + bottom_e2

        top_e3 = self.residual3(e2) #32*32*256
        bottom_e3 = self.asymmetric3(e2) #32*32*256
        e3 = top_e3 + bottom_e3

        top_e4 = self.residual4(e3) #16*16*512
        bottom_e4 = self.asymmetric4(e3) #16*16*512
        e4 = top_e4 + bottom_e4 #16*16*1024
        

        # Decoder
        d4 = self.decoder4(e4) + e3 #32*32*512
        d3 = self.decoder3(d4) + e2 #64*64*256
        d2 = self.decoder2(d3) + e1 #128*128*128
        d1 = self.decoder1(d2) #256*256*64
        out = self.finaldeconv1(d1) #513*513*32
        out = self.finalrelu1(out)
        out = self.finalconv2(out) #511*511*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) #512*512*32

        return torch.sigmoid(out) #512*512*1

def DBRANet_2():
    return DBRANet(pool_size = 2)

def DBRANet_4():
    return DBRANet(pool_size = 4)

def DBRANet_8():
    return DBRANet(pool_size = 8)
    
def DBRANet_PW_2():
    return DBRANet_PW(pool_size = 2)

def DBRANet_PW_4():
    return DBRANet_PW(pool_size = 4)

def DBRANet_PW_8():
    return DBRANet_PW(pool_size = 8)

def DBRANet_RAD2():
    return DBRANet_RAD(pool_size = 2)

def DBRANet_RAD4():
    return DBRANet_RAD(pool_size = 4)

def DBRANet_RAD8():
    return DBRANet_RAD(pool_size = 8)
"""
if __name__ == '__main__':

    img = torch.randn(2, 3, 512, 512)
    # net = RefinedAsymmetricBlock()._make_layer(BasicAsymmetricBlock,64,3,stride=2)
    # net = DBRANet_4()
    # out = net(img)
    # print(out.size())
    # flops, params = thop.profile(net, inputs=(img, ))
    # print('Params: %.2f M' % (params/(1000)**2))
"""
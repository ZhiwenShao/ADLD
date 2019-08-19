import torch
import torch.nn as nn
import torch.nn.functional as F


class Feat_Enc(nn.Module):
    def __init__(self):
        super(Feat_Enc, self).__init__()

        self.align_attention_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Tanh(),
        )

    def forward(self, x):
        align_output = self.align_attention_features(x)

        return align_output


class AU_Detect(nn.Module):
    def __init__(self, au_num):
        super(AU_Detect, self).__init__()

        self.aus_feat = nn.ModuleList([nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        ) for i in range(au_num)])

        self.aus_fc = nn.ModuleList([
            nn.Linear(64, 1)
            for i in range(au_num)])

    def forward(self, x):
        start = True
        for i in range(len(self.aus_fc)):
            au_feat = self.aus_feat[i](x)

            au_feat_interm = F.avg_pool2d(au_feat, au_feat.size()[2:])
            au_feat_interm = au_feat_interm.view(au_feat_interm.size(0), -1)
            au_output = self.aus_fc[i](au_feat_interm)

            if start:
                aus_output = au_output
                aus_feat = au_feat_interm
                start = False
            else:
                aus_output = torch.cat((aus_output, au_output), 1)
                aus_feat = torch.cat((aus_feat, au_feat_interm), 1)

        return aus_feat, aus_output


class Land_Detect(nn.Module):
    def __init__(self, land_num):
        super(Land_Detect, self).__init__()

        self.align_attention_features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
            nn.Tanh(),

            nn.Conv2d(64, land_num, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        align_feat = self.align_attention_features[:-1](x)
        align_output = self.align_attention_features[-1](align_feat)

        start = True
        for i in range(align_output.size(1)):
            t_align_attention_feat_ori = align_output[:, i, :, :]
            t_align_attention_feat = t_align_attention_feat_ori.view(t_align_attention_feat_ori.size(0), -1)
            t_align_attention_feat = F.softmax(t_align_attention_feat, 1)
            t_align_attention_feat = t_align_attention_feat.view(t_align_attention_feat_ori.size(0), 1,
                                                                 t_align_attention_feat_ori.size(1),
                                                                 t_align_attention_feat_ori.size(2))
            if start:
                align_attention = t_align_attention_feat
                start = False
            else:
                align_attention = torch.cat((align_attention, t_align_attention_feat), 1)

        return align_attention, align_feat, align_output


class Texture_Enc(nn.Module):
    def __init__(self, inter_dim=64):
        super(Texture_Enc, self).__init__()

        self.irrele_shape_encoder = nn.Sequential(
            nn.Conv2d(64, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
            nn.Tanh(),
        )

    def forward(self, x):
        irrele_shape_output = self.irrele_shape_encoder(x)
        return irrele_shape_output


class Generator(nn.Module):
    def __init__(self, input_dim1 = 1, input_dim2=64, inter_dim=128):
        super(Generator, self).__init__()

        self.feat_generator = nn.Sequential(
            nn.Conv2d(input_dim1 + input_dim2, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim // 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim // 2, inter_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(inter_dim // 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim // 2, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
            nn.Tanh(),
        )

    def forward(self, align_attentions, irrele_shape_output):
        assemble_align_attention = torch.sum(align_attentions, 1, True)
        input = torch.cat((assemble_align_attention, irrele_shape_output), 1)
        # input = torch.cat((align_attentions,irrele_shape_output),1)
        output = self.feat_generator(input)
        return output


class Land_Disc(nn.Module):
    def __init__(self, land_num):
        super(Land_Disc, self).__init__()

        self.align_attention_features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, land_num, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        align_output = self.align_attention_features(x)

        return align_output


class Discriminator(nn.Module):
    '''Discriminator model for source domain.'''

    def __init__(self, input_dim=64, inter_dim = 64):
        '''Init discriminator.'''
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, inter_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(inter_dim),
            nn.PReLU(),

            nn.Conv2d(inter_dim, inter_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(inter_dim * 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 2, inter_dim * 2, kernel_size=4, stride=2, padding=0),
            # nn.InstanceNorm2d(inter_dim * 2),
            nn.PReLU(),

            nn.Conv2d(inter_dim * 2, 1, kernel_size=1, stride=1, padding=0)
        )
        self.input_dim = input_dim

    def forward(self, input):

        out = self.layer(input)
        out = out.view(out.size(0), -1)
        return out


network_dict = {'Feat_Enc':Feat_Enc, 'Land_Detect':Land_Detect, 'AU_Detect':AU_Detect, 'Land_Disc':Land_Disc,
                'Texture_Enc':Texture_Enc, 'Generator':Generator, 'Discriminator':Discriminator}
import torch.nn as nn
from .sfenet import SfeNet
from .bert import Bert
import torchvision.models as models
import torch
import pickle


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.part2 = args.part2
        self.part3 = args.part3

        self.image_model = SfeNet()  
        self.language_model = Bert()
        
        inp_size = 2048
    
        # shorten the tensor using 1*1 conv
        self.conv_images = nn.Conv2d(inp_size, args.feature_size, 1)
        self.conv_text = nn.Conv2d(768, args.feature_size, 1)

        # BN layer before embedding projection
        self.bottleneck_image = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)

        self.bottleneck_text = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)

        self.local_fc_text_key = nn.Linear(768, args.feature_size)
        self.local_bottleneck_text_key = nn.LayerNorm([98 + 2 + 1, args.feature_size])

        self.local_fc_text_value = nn.Linear(768, args.feature_size)
        self.local_bottleneck_text_value = nn.LayerNorm([98 + 2 + 1, args.feature_size])

        self.global_image_query = nn.Linear(args.feature_size, args.feature_size)
        self.global_image_value = nn.Linear(args.feature_size, args.feature_size)


        self.fc_p2_list = nn.ModuleList([nn.Linear(inp_size, args.feature_size) for i in range(self.part2)])
        self.fc_p3_list = nn.ModuleList([nn.Linear(inp_size, args.feature_size) for i in range(self.part3)])

        self.fc_p2_list_query = nn.ModuleList([nn.Linear(args.feature_size, args.feature_size) for i in range(self.part2)])
        self.fc_p2_list_value = nn.ModuleList([nn.Linear(args.feature_size, args.feature_size) for i in range(self.part2)])

        self.fc_p3_list_query = nn.ModuleList([nn.Linear(args.feature_size, args.feature_size) for i in range(self.part3)])
        self.fc_p3_list_value = nn.ModuleList([nn.Linear(args.feature_size, args.feature_size) for i in range(self.part3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, images, tokens, segments, input_masks, sep_tokens, sep_segments, sep_input_masks, n_sep, p2=None, p3=None, object=None, attribute=None, stage=''):

        text_features = self.language_model(sep_tokens, sep_segments, sep_input_masks)

        local_text_feat = text_features[:, 0, :]
        local_text_feat = local_text_feat.view(-1, n_sep, local_text_feat.size(1))

        b1, part_feature_list_b2, part_feature_list_b3 = self.image_model(images, p2, p3)
        text_features = self.language_model(tokens, segments, input_masks)

        global_img_feat, global_text_feat = self.build_joint_embeddings(b1, text_features[:, 0])
        global_img_feat = self.bottleneck_image(global_img_feat)
        global_text_feat = self.bottleneck_text(global_text_feat)

        local_text_feat = torch.cat((global_text_feat.unsqueeze(1), local_text_feat, text_features[:, 1:99]), dim = 1)
        local_text_key = self.local_fc_text_key(local_text_feat)
        local_text_key = self.local_bottleneck_text_key(local_text_key)

        local_text_value = self.local_fc_text_value(local_text_feat)
        local_text_value = self.local_bottleneck_text_value(local_text_value)

        for i in range(len(part_feature_list_b2)):
            part_feature_list_b2[i] = self.fc_p2_list[i](part_feature_list_b2[i])
        for i in range(len(part_feature_list_b3)):
            part_feature_list_b3[i] = self.fc_p3_list[i](part_feature_list_b3[i])

            
        global_img_query = self.global_image_query(global_img_feat)
        global_img_value = self.global_image_value(global_img_feat)

        local_img_query = torch.zeros(global_img_feat.shape[0], self.part2 + self.part3, global_img_feat.shape[1]).cuda()
        local_img_value = torch.zeros(global_img_feat.shape[0], self.part2 + self.part3, global_img_feat.shape[1]).cuda()

        for i in range(len(part_feature_list_b2)):
            local_img_query[:, i, :] = self.fc_p2_list_query[i](part_feature_list_b2[i])
        for i in range(len(part_feature_list_b3)):
            local_img_query[:, i + self.part2, :] = self.fc_p3_list_query[i](part_feature_list_b3[i])

        for i in range(len(part_feature_list_b2)):
            local_img_value[:, i, :] = self.fc_p2_list_value[i](part_feature_list_b2[i])
        for i in range(len(part_feature_list_b3)):
            local_img_value[:, i + self.part2, :] = self.fc_p3_list_value[i](part_feature_list_b3[i])

        local_img_query = torch.cat((global_img_query.unsqueeze(1), local_img_query), dim = 1)
        local_img_value = torch.cat((global_img_value.unsqueeze(1), local_img_value), dim = 1)

        return global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value


    def build_joint_embeddings(self, images_features, text_features):

        text_features = text_features.unsqueeze(2)
        text_features = text_features.unsqueeze(3)
        
        image_embeddings = self.conv_images(images_features).squeeze()
        text_embeddings = self.conv_text(text_features).squeeze()

        return image_embeddings, text_embeddings



















import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle

from torch.autograd import Variable

logger = logging.getLogger()                                                                                                                                                                            
logger.setLevel(logging.INFO)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def compute_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def pairwise_distance(A, B):
    """
    Compute distance between points in A and points in B
    :param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
    :param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
    :return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
    """
    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)

    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

    return distance


def func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, opt, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    opt: parameters
    """
    batch_size, queryL, sourceL = txt_i_key_expand.size(
        0), local_img_query.size(1), txt_i_key_expand.size(1)
    local_img_query_norm = l2norm(local_img_query, dim=-1)
    txt_i_key_expand_norm = l2norm(txt_i_key_expand, dim=-1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    local_img_queryT = torch.transpose(local_img_query_norm, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(txt_i_key_expand_norm, local_img_queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * opt.lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # print('attn: ', attn)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if opt.focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif opt.focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        funcH = None
    

    # Step 3: reassign attention
    if funcH is not None:
        tmp_attn = funcH * attn
        attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
        attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    txt_i_valueT = torch.transpose(txt_i_value_expand, 1, 2)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(txt_i_valueT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    #return weightedContext, attn
    return weightedContext

def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size * queryL, sourceL, 1)
    xj = xj.view(batch_size * queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size * queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1 - term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def constraints(features, labels):
    labels = torch.reshape(labels, (labels.shape[0],1))
    con_loss = AverageMeter()
    index_dict = {k.item() for k in labels}
    for index in index_dict:
        labels_mask = (labels == index)
        feas = torch.masked_select(features, labels_mask)
        feas = feas.view(-1, features.shape[1])
        distance = pairwise_distance(feas, feas)
        #torch.sqrt_(distance)
        num = feas.shape[0] * (feas.shape[0] - 1)
        loss = torch.sum(distance) / num
        con_loss.update(loss, n = num / 2)
    return con_loss.avg


def constraints_loss(data_loader, network, args):
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size,args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    con_images = 0.0
    con_text = 0.0
    with torch.no_grad():
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index: index + interval] = labels
            index = index + interval
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
    
    if args.constraints_text:
        con_text = constraints(text_bank, labels_bank)
    if args.constraints_images:
        con_images = constraints(images_bank, labels_bank)

    return con_images, con_text
   

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.CMPM = args.CMPM
        self.CMPC = args.CMPC
        self.CONT = args.CONT
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        if args.resume:
            checkpoint = torch.load(args.model_path)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    @staticmethod
    def compute_weiTexts(local_img_query, local_img_value, local_text_key, local_text_value, text_length, args):
        """
        Compute weighted text embeddings
        :param image_embeddings: Tensor with dtype torch.float32, [n_img, n_region, d]
        :param text_embeddings: Tensor with dtype torch.float32, [n_txt, n_word, d]
        :param text_length: list, contain length of each sentence, [batch_size]
        :param labels: Tensor with dtype torch.int32, [batch_size]
        :return: i2t_similarities: Tensor, [n_img, n_txt]
                 t2i_similarities: Tensor, [n_img, n_txt]
        """
        n_img = local_img_query.shape[0]
        n_txt = local_text_key.shape[0]
        t2i_similarities = []
        i2t_similarities = []
        #atten_final_result = {}
        for i in range(n_txt):
            # Get the i-th text description
            n_word = text_length[i]
            txt_i_key = local_text_key[i, :n_word, :].unsqueeze(0).contiguous()
            txt_i_value = local_text_value[i, :n_word, :].unsqueeze(0).contiguous()
            # -> (n_img, n_word, d)
            txt_i_key_expand = txt_i_key.repeat(n_img, 1, 1)
            txt_i_value_expand = txt_i_value.repeat(n_img, 1, 1)

            # -> (n_img, n_region, d)
            #weiText, atten_text = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
            weiText = func_attention_MxN(local_img_query, txt_i_key_expand, txt_i_value_expand, args)
            #atten_final_result[i] = atten_text[i, :, :]
            # image_embeddings = l2norm(image_embeddings, dim=2)
            weiText = l2norm(weiText, dim=2)
            i2t_sim = compute_similarity(local_img_value, weiText, dim=2)
            i2t_sim = i2t_sim.mean(dim=1, keepdim=True)
            i2t_similarities.append(i2t_sim)

            # -> (n_img, n_word, d)
            #weiImage, atten_image = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
            weiImage = func_attention_MxN(txt_i_key_expand, local_img_query, local_img_value, args)
            # txt_i_expand = l2norm(txt_i_expand, dim=2)
            weiImage = l2norm(weiImage, dim=2)
            t2i_sim = compute_similarity(txt_i_value_expand, weiImage, dim=2)
            t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
            t2i_similarities.append(t2i_sim)

        # (n_img, n_txt)
        #torch.save(atten_final_result, 'atten_final_result.pt')
        i2t_similarities = torch.cat(i2t_similarities, 1)
        t2i_similarities = torch.cat(t2i_similarities, 1)

        return i2t_similarities, t2i_similarities

    def contrastive_loss(self, i2t_similarites, t2i_similarities, labels):
        batch_size = i2t_similarites.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        criterion = nn.CrossEntropyLoss()

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
       

        i2t_pred = F.softmax(i2t_similarites * self.args.lambda_softmax, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(i2t_similarites * self.args.lambda_softmax, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        sim_cos = i2t_similarites

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        # constrastive_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        constrastive_loss = torch.mean(torch.sum(i2t_loss, dim=1))

        return constrastive_loss, pos_avg_sim, neg_avg_sim


    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = F.normalize(self.W, p=2, dim=0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)

        # image_logits = torch.matmul(image_embeddings, self.W_norm)
        # text_logits = torch.matmul(text_embeddings, self.W_norm)

        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]

        # print("batch size: " + str(batch_size))

        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        # i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        t2i_pred = F.softmax(text_proj_image, dim=1)
        # t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def forward(self, global_img_feat, global_text_feat, local_img_query, local_img_value, local_text_key, local_text_value, text_length,
                labels):
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        cont_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim = 0.0
        local_pos_avg_sim = 0.0
        local_neg_avg_sim = 0.0
        if self.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(global_img_feat, global_text_feat,
                                                                         labels)
        if self.CMPC:
            cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(global_img_feat,
                                                                                global_text_feat, labels)
        if self.CONT:
            i2t_sim, t2i_sim = self.compute_weiTexts(local_img_query, local_img_value, local_text_key, local_text_value, text_length, self.args)
            cont_loss, local_pos_avg_sim, local_neg_avg_sim = self.contrastive_loss(i2t_sim, t2i_sim, labels)
            cont_loss = cont_loss * self.args.lambda_cont

        loss = cmpm_loss + cmpc_loss + cont_loss

        return cmpm_loss.item(), cmpc_loss.item(), cont_loss.item(), loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim, local_pos_avg_sim, local_neg_avg_sim


class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count


def compute_topk(query_global, query, value_bank, gallery_global, gallery_key, gallery_value,
                       gallery_length, target_query, target_gallery, args, k_list=[1, 5, 20], reverse=False):
    global_result = []
    local_result = []
    result = []
    sim_cosine = []

    query_global = F.normalize(query_global, p=2, dim=1)
    gallery_global = F.normalize(gallery_global, p=2, dim=1)

    sim_cosine_global = torch.matmul(query_global, gallery_global.t())

    
    for i in range(0, query.shape[0], 200):
        i2t_sim, t2i_sim = Loss.compute_weiTexts(query[i:i + 200], value_bank[i:i + 200], gallery_key, gallery_value, gallery_length, args)
        sim = i2t_sim
        sim_cosine.append(sim)

    sim_cosine = torch.cat(sim_cosine, dim=0)
    
    sim_cosine_all = sim_cosine_global + sim_cosine
    reid_sim = None
    if(args.reranking):
        reid_sim = torch.matmul(query_global, query_global.t())

    global_result.extend(topk(sim_cosine_global, target_gallery, target_query, k=k_list))
    if reverse:
        global_result.extend(topk(sim_cosine_global, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    local_result.extend(topk(sim_cosine, target_gallery, target_query, k=k_list))
    if reverse:
        local_result.extend(topk(sim_cosine, target_query, target_gallery, k=k_list, dim=0, print_index=False))

    result.extend(topk(sim_cosine_all, target_gallery, target_query, k=k_list, reid_sim=reid_sim))
    if reverse:
        result.extend(topk(sim_cosine_all, target_query, target_gallery, k=k_list, dim=0, print_index=False, reid_sim=reid_sim))
    return global_result, local_result, result

def jaccard(a_list,b_list):
    return 1.0 - float(len(set(a_list)&set(b_list)))/float(len(set(a_list)|set(b_list)))*1.0

def topk(sim, target_gallery, target_query, k=[1,5,10], dim=1, print_index=False, reid_sim = None):
    result = []
    maxk = max(k)
    size_total = len(target_query)
    if reid_sim is None:
        _, pred_index = sim.topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
    else:
        K = 5
        sim = sim.cpu().numpy()
        reid_sim = reid_sim.cpu().numpy()
        pred_index = np.argsort(-sim, axis = 1)
        reid_pred_index = np.argsort(-reid_sim, axis = 1)

        q_knn = pred_index[:, 0:K]
        g_knn = reid_pred_index[:, 0:K]

        new_index = []
        jaccard_dist = np.zeros_like(sim)
        from scipy.spatial import distance
        for i, qq in enumerate(q_knn):
            for j, gg in enumerate(g_knn):
                jaccard_dist[i, j] = 1.0 - jaccard(qq, gg)
        _, pred_index = torch.Tensor(sim + jaccard_dist).topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
  

    # pred
    if dim == 1:
        pred_labels = pred_labels.t()

    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))
    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result

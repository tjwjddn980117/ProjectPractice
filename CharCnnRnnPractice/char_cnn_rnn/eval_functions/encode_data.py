import os
import torch
import torchfile

import char_cnn_rnn as ccr
import ccr_utils

def encode_data(net_txt, net_img, data_dir, split, num_txts_eval, batch_size, device):
    '''
    Encoder for preprocessed Caltech-UCSD Birds 200-2011 and Oxford 102
    Category Flowers datasets, used in ``Learning Deep Representations of
    Fine-grained Visual Descriptions``.

    Warning: if you decide to not use all sentences (i.e., num_txts_eval > 0),
    sentences will be randomly sampled and their features will be averaged to
    provide a class representation. This means that the evaluation procedures
    should be performed multiple times (using different seeds) to account for
    this randomness.
    
    Arguments:
        net_txt (torch.nn.Module): text processing network.
        net_img (torch.nn.Module): image processing network.
        data_dir (string): path to directory containing dataset files.
        split (string): which data split to load.
        num_txts_eval (int): number of textual descriptions to use for each
            class (0 = use all). The embeddings are averaged per-class.
        batch_size (int): batch size to split data processing into chunks.
        device (torch.device): which device to do computation in.

    Returns:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.
    '''

    # for example 'valclasses.txt', 'textclasses.txt'
    path_split_file = os.path.join(data_dir, split+'classes.txt')
    cls_list = [line.rstrip('\n') for line in open(path_split_file)]

    cls_feats_img = []
    cls_feats_txt = []
    for cls in cls_list:
        '''
        prepare image data
        '''
        data_img_path = os.path.join(data_dir, 'images', cls + '.t7')
        data_img = torch.Tensor(torchfile.load(data_img_path))
        # cub and flowers datasets have 10 image crops per instance
        # we use only the first crop per instance
        feats_img = data_img[:, :, 0].to(device)
        if net_img is not None: # if we had net_img
            with torch.no_grad(): # extracting feature image with net_image
                feats_img = net_img(feats_img)
        # one cls_feature_image for one class
        cls_feats_img.append(feats_img)


        '''
        prepare image data    
        '''
        data_txt_path = os.path.join(data_dir, 'text_c10', cls + '.t7')
        data_txt = torch.LongTensor(torchfile.load(data_txt_path))

        # size of data_txt was [num_of_instance, num_of_lenght, num_of_description]
        # select T texts from all instances to represent this class
        data_txt = data_txt.permute(0, 2, 1)
        # after permute, data_tax will [num_of_inst, num_of_descript, num_of_len]

        total_txts = data_txt.size(0) * data_txt.size(1)
        # 2d flatten
        # it makes data_txt to [num_of_inst, descriptions]
        data_txt = data_txt.contiguous().view(total_txts, -1)

        if num_txts_eval > 0:
            num_txts_eval = min(num_txts_eval, total_txts)
            id_txts = torch.randperm(data_txt.size(0))[:num_txts_eval]
            data_txt = data_txt[id_txts]
            # [num_txts_eval, descriptions]

        # convert to one-hot tensor to run through network
        # TODO: adapt code to support batched version
        txt_onehot = []
        for txt in data_txt: # txt is descriptions
            # size of output 'ccr_utils.labelvec_to_onehot' is [emb_size, max_str_len]
            # so, we append [emb_size, max_str_len]
            txt_onehot.append(ccr_utils.labelvec_to_onehot(txt))
        # size of txt_onehot is [num_txts_eval, emb_size, max_str_len]
        txt_onehot = torch.stack(txt_onehot)

        # if we use a lot of text descriptions, it will not fit in gpu memory
        # separate instances into mini-batches to process them using gpu
        feats_txt = []
        for batch in torch.split(txt_onehot, batch_size, dim=0):
            with torch.no_grad():
                # batch will [batch_size, emb_size, max_str_len]
                out = net_txt(batch.to(device))
                # output will be [batch_size, 1024] (embedding)
            feats_txt.append(out)
        
        # average the outputs
        # torch.cat should return [full_batch, 1024] 
        # then, mean(dim=0) make feats_txt to [1024, ]
        feats_txt = torch.cat(feats_txt, dim=0).mean(dim=0)
        # so, [1024, ] means that embedding of all of seleceted instance
        cls_feats_txt.append(feats_txt)

    # embedding with all classes
    # size of cls_feats_txt is [num_class, 1024]
    cls_feats_txt = torch.stack(cls_feats_txt, dim=0)

    # cls_feats_img is [num_class, feats_img, 1024]
    # cls_feats_txt is [num_class, 1024]
    # cls_list is [num_class, ]
    return cls_feats_img, cls_feats_txt, cls_list
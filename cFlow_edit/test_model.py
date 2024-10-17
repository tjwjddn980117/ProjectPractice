import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.dataset import LIDC
from data.lidc_our_dataset import lidc_Dataloader
from data.dataset_config import exp_config
from models.cflownet import cFlowNet
import torch.nn as nn
from models.unet import Unet
from utils.utils import l2_regularisation,ged
from utils.save_results_1channel import save_results, save_results_with_batch
import time
from utils.tools import makeLogFile, writeLog, dice_loss
import pdb
import argparse
import sys
import os
from PIL import Image
import multiprocessing
from tqdm import tqdm

def main():
    # torch.manual_seed(42)
    # np.random.seed(42)


    parser = argparse.ArgumentParser()
    parser.add_argument('--flow', action='store_true', default=False, help=' Train with Flow model')
    parser.add_argument('--glow', action='store_true', default=False, help=' Train with Glow model')
    parser.add_argument('--data', type=str, default='data/lidc/',help='Path to data.')
    parser.add_argument('--probUnet', action='store_true', default=False, help='Train with Prob. Unet')
    parser.add_argument('--unet', action='store_true', default=False, help='Train with Det. Unet')
    parser.add_argument('--singleRater', action='store_true', default=False, help='Train with single rater')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    # parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_flows', type=int, default=4, help='Num flows')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = LIDC(data_dir=args.data)
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(0.2 * dataset_size))
    # 
    # np.random.shuffle(indices)
    # valid_indices, test_indices, train_indices = indices[:split], indices[split:2*split], indices[2*split:]
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(valid_indices)
    # test_sampler = SubsetRandomSampler(test_indices)
    # train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    # valid_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)
    # test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

    lidc = lidc_Dataloader(exp_config)

    train_loader = lidc.train
    valid_loader = lidc.validation
    test_loader = lidc.test

    #print("Number of train/valid/test patches:", (len(train_indices),len(valid_indices),len(test_indices)))

    fName = time.strftime("%Y%m%d_%H_%M")

    if args.singleRater:
        print("Using a single rater..")
        fName = fName+'_1R'
    else:
        print("Using all experts...")
        fName = fName+'_4R'

    if args.flow:
        print("Using flow based model with %d steps"%args.num_flows)
        fName = fName+'_flow'
        net = cFlowNet(input_channels=1, num_classes=1, 
    			num_filters=[32,64,128,256], latent_dim=6, 
            	no_convs_fcomb=4, num_flows=args.num_flows, 
    			norm=True,flow=args.flow)
    elif args.glow:
        print("Using Glow based model with %d steps"%args.num_flows)
        fName = fName+'_glow'
        net = cFlowNet(input_channels=1, num_classes=1,
    			num_filters=[32,64,128,256], latent_dim=6, 
    			no_convs_fcomb=4, norm=True,num_flows=args.num_flows,
    			flow=args.flow,glow=args.glow)
    elif args.probUnet:
        print("Using probUnet")
        fName = fName+'_probUnet'
        net = cFlowNet(input_channels=1, num_classes=1, 
    			num_filters=[32,64,128,256], latent_dim=6, 
            	no_convs_fcomb=4, norm=True,flow=args.flow)
    elif args.unet:
        print("Using Det. Unet")
        fName = fName+'_Unet'
        net = Unet(input_channels=1, num_classes=1, 
    			num_filters=[32,64,128,256], apply_last_layer=True, 
                padding=True, norm=True, 
    			initializers={'w':'he_normal', 'b':'normal'})
        criterion = nn.BCELoss(size_average=False)
    else:
        print("Choose a model.\nAborting....")
        sys.exit()

    if not os.path.exists('logs'):
        os.mkdir('logs')

    logFile = 'logs/'+fName+'.txt'
    makeLogFile(logFile)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    nTrain = len(train_loader)
    nValid = len(valid_loader)
    nTest = len(test_loader)

    minLoss = 1e8

    convIter=0
    convCheck = 20

    load_model_path = 'C:/Users/Seo/Downloads/CFlow_/20241013_15_43_4R_glow.pt'
    net.load_state_dict(torch.load(load_model_path))
    net.eval()

    test_loss = []
    dices = []

    print("Starting testing...")
    #print(test_loader)
    reconstruction_results = None
    with torch.no_grad():
        #print("start")
        path_list = []
        for step, (patch, masks, pred_prob, path) in enumerate(tqdm(test_loader)):
            #if step == 0:
            #    print("hello")
            #    print(path)
            patch = patch.to(device)
            masks = masks.to(device)
            if args.singleRater or args.unet:
                rater = 0
            else:
                rater = torch.randperm(4)[0]
            mask = masks[:, [rater]]
            path_list.append(path)
            if not args.unet:
                # Use the probabilistic model for inference
                net.forward(patch, mask, training=True)
                comb, _, _ = net.sample()
                
                # # Convert reconstruction to tensor if needed
                # reconstruction, recLoss, kl, elbo = net.elbo(mask, use_mask=False, analytic_kl=False)
                # reconstruction = torch.sigmoid(reconstruction).squeeze(1).cpu().numpy()  # [batch, H, W]
                # reconstruction_tensor = torch.tensor(reconstruction)

                # Generate 16 samples for each patch
                samples = []
                for _ in range(16):
                    # Sampling with ELBO and apply sigmoid to convert logits to probabilities
                    #reconstruction, recLoss, kl, elbo = net.elbo(mask, use_mask=False, analytic_kl=False)
                    reconstruction, _, _ = net.sample()
                    reconstruction = torch.sigmoid(reconstruction).squeeze(1)  # [batch, H, W]
                    samples.append(reconstruction)
            
                # Stack all 16 samples into a single tensor
                samples_tensor = torch.stack(samples, dim=1)  # [batch_size, 16, H, W]
                samples_tensor = samples_tensor.cpu().numpy()  # Convert to numpy array if needed
                samples_tensor = torch.tensor(samples_tensor)  # Back to tensor for further processing
                
                save_results_with_batch(samples_tensor, path, 'C:/Users/Seo/Downloads/cFlow-master/Results_3')
                #differences = []
                #for i in range(1, 16):
                #    diff = torch.abs(samples_tensor[:, 0] - samples_tensor[:, i])
                #    differences.append(diff.mean().item())

                # Concatenate results
                # if reconstruction_results is None:
                #     reconstruction_results = samples_tensor  
                # else:
                #     reconstruction_results = torch.cat((reconstruction_results, samples_tensor), dim=0)  
                #if step < 5:
                #    print(f'Mean differences between samples: {sum(differences) / len(differences)}')
                #    print(f'samples_shape: {samples_tensor.shape}')

            else:
                pred = torch.sigmoid(net.forward(patch, False))
                loss = criterion(target=mask, input=pred)
                if step == 0:
                    print()
                    print(f'pred_shape: {pred.shape}')
        
        print('finish!')
            

    #print(f'reconstruction_results_shape: {reconstruction_results.shape}')
    #save_results(reconstruction_results, path_list, 'C:/Users/Seo/Downloads/cFlow-master/Results')
    #output_dir = 'C:/Users/Seo/Downloads/cFlow-master/outputs'
    #os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(output_dir+'/numpy', exist_ok=True)
    #os.makedirs(output_dir+'/segment_sample', exist_ok=True)
#
    ## Reconstruction 결과를 numpy로 저장
    #np.save(os.path.join(output_dir, 'numpy', 'reconstruction_results.npy'), reconstruction_results.numpy())
#
    ## 각각의 reconstruction 이미지를 저장
    #for i in range(reconstruction_results.shape[0]):
    #    # 각 이미지를 [H, W] 형식으로 변환
    #    img = reconstruction_results[i]  # [H, W]
    #    img = img.numpy()  # Tensor를 numpy 배열로 변환
    #    img = (img * 255).astype(np.uint8)  # 0-255 범위로 변환하여 uint8 타입으로 변환
    #    img_path = os.path.join(output_dir, 'segment_sample', f'reconstruction_{i}.png')
    #    Image.fromarray(img).save(img_path)
#
    #print("Reconstruction results saved successfully.")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # 프로그램이 실행 파일로 패키징될 경우
    main()
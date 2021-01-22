import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import scipy
from scipy.io import savemat
from torchvision import transforms
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from NCSN_2C_train.runners_SIAT.python_L2_ifer import L2_image_from_edges_rect,L2_image_from_edges_rect_reverse
from NCSN_2C_train.models.cond_refinenet_dilated import CondRefineNetDilated

__all__ = ['siat_multicontrast_compare_TSE_sag_random_R4_GS']

def compare_snr(sig, ref):
    sig = abs(sig)
    ref = abs(ref)
    mse = np.mean( (sig.flatten() - ref.flatten()) ** 2) 
    dv = np.var(ref.flatten(),axis = 0)
    x = 10*math.log10(dv/mse)
    
    return x

def compare_rmse(sig, ref):
    x1 = sig.flatten()-ref.flatten()
    x2 = ref.flatten()
    y = np.linalg.norm(x1)/np.linalg.norm(x2)
    
    return y
    
# show image
def show(image):
    plt.figure(1)
    plt.imshow(np.abs(image),cmap='gray',vmax=1,vmin=0)
    plt.show()

class siat_multicontrast_compare_TSE_sag_random_R4_GS():
    def __init__(self, args, config):
        self.args = args
        self.config = config
    # test function
    def test(self):

        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint_40000.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        dataset_name = 'Sag_SlcRaw4'

        result_all = np.zeros([1,3])
    
        # prepare test data
        data = np.zeros([256,256,3]).astype(np.complex64)
        data[:,:,0] = loadmat('./Sag_Multicontrast/PD_Sag_SlcRaw4.mat')['Img'].astype(np.complex64)
        data[:,:,1] = loadmat('./Sag_Multicontrast/T1_Sag_SlcRaw4.mat')['Img'].astype(np.complex64)
        data[:,:,2] = loadmat('./Sag_Multicontrast/T2_Sag_SlcRaw4.mat')['Img'].astype(np.complex64)
        # normalization
        data[:,:,0] = (np.real(data[:,:,0]) - np.min(abs(data[:,:,0]))+1j*np.imag(data[:,:,0]))/(np.max(abs(data[:,:,0]))-np.min(abs(data[:,:,0])))
        data[:,:,1] = (np.real(data[:,:,1]) - np.min(abs(data[:,:,1]))+1j*np.imag(data[:,:,1]))/(np.max(abs(data[:,:,1]))-np.min(abs(data[:,:,1])))
        data[:,:,2] = (np.real(data[:,:,2]) - np.min(abs(data[:,:,2]))+1j*np.imag(data[:,:,2]))/(np.max(abs(data[:,:,2]))-np.min(abs(data[:,:,2])))
        data_concat = np.concatenate((data[:,:,0], data[:,:,1], data[:,:,2]),axis = 1)
        # save the fully sampled image
        cv2.imwrite(os.path.join(self.args.image_folder, 'sdata.png' ),(data_concat*255).astype(np.uint8))
        data2 = data.transpose(2,0,1)
        # get the undersample mask
        mask = loadmat('./mask/random_mask_256_R4_BCS_ours_Sag.mat')['random_mask_256_R4_BCS_ours_Sag']#.astype(np.complex64)
        for i in range(3):
            mask[:,:,i]=np.fft.fftshift(mask[:,:,i])

        kdata = np.zeros([256,256,3]).astype(np.complex64)
        
        for i in range(3): 
            kdata[:,:,i]=np.fft.fft2(data[:,:,i])
        print('value max min :',np.max(data),np.min(data))
        # get the undersampled k-space data
        ksample=np.multiply(mask,kdata)

        ksample = ksample.transpose(2,0,1)
        x0 = nn.Parameter(torch.Tensor(3,4,256,256).uniform_(-1,1)).cuda()
        x01 = x0
        x02 = x0
        x03 = x0

        step_lr= 0.5*0.0001   

        # noise amounts
        sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
        # set parameters
        n_steps_each = 40
        max_psnr = 0
        max_snr = 0
        min_rmse = 1
        m = np.shape(data)[0]
        n = np.shape(data)[1]
        betaliu = 0.001
        
        for idx, sigma in enumerate(sigmas):
            print(idx)
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            print('sigma = {}'.format(sigma))
            # the inner iteration loop
            for step in range(n_steps_each):
                # gradient domain prior update by ncsn
                noise1 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                noise2 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                noise3 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                grad1 = scorenet(x01, labels).detach()
                grad2 = scorenet(x02, labels).detach()
                grad3 = scorenet(x03, labels).detach()

                x0 = x0 + step_size * (grad1 + grad2 + grad3)/3.0 
                
                x01 = x0 + noise1
                x02 = x0 + noise2
                x03 = x0 + noise3
                                            
                x0=np.array(x0.cpu().detach(),dtype = np.float32)
                
                x_complex = np.zeros([3,256,256]).astype(np.complex64)

                for imageindex in range(3):
                    x_complex[imageindex,:,:] = L2_image_from_edges_rect(ksample[imageindex,:,:],x0[imageindex,0,:,:]+1j*x0[imageindex,1,:,:],x0[imageindex,2,:,:]+1j*x0[imageindex,3,:,:],0)
                
                x0 = x0.transpose(0,2,3,1)
                for imageindex in range(3):
                    x0[imageindex,:,:,:] = L2_image_from_edges_rect_reverse(x_complex[imageindex,:,:])
                x0 = x0.transpose(0,3,1,2)
                
                # group sparsity
                if self.args.GS:
                    x0_vector = np.reshape(x0, (m*n, 12))

                    temp1 = np.sqrt(np.sum(x0_vector**2,axis = 1))[:,np.newaxis] 
                    temp2 = np.maximum(0, 1-betaliu/temp1)
   
                    x0_vector = np.tile(temp2,[1,12])*x0_vector
                    x0 = np.reshape(x0_vector,(3,4,m,n))
                
                x0 = torch.tensor(x0,dtype=torch.float32).cuda()
                # get rec image and err image
                err0 = data2[0,:,:] - x_complex[0,:,:]
                err1 = data2[1,:,:] - x_complex[1,:,:]
                err2 = data2[2,:,:] - x_complex[2,:,:]
                err_TSE_concat = np.concatenate((err0, err1,err2),axis = 1)  
                x_complex_concat = np.concatenate((x_complex[0,:,:], x_complex[1,:,:], x_complex[2,:,:]),axis = 1)

                # evaluation indicators
                psnr = compare_psnr(255*abs(x_complex_concat.flatten()),255*abs(data_concat.flatten()),data_range=255)
                snr = compare_snr(255*abs(x_complex_concat.flatten()),255*abs(data_concat.flatten()))                                   
                rmse = compare_rmse(255*abs(x_complex_concat.flatten()),255*abs(data_concat.flatten()))
                print("current {} step".format(step),'PSNR :', psnr,'RMSE :', rmse,'SNR :', snr)

                if max_psnr < psnr :
                    result_all[0,0] = psnr
                    max_psnr = psnr
                    # save rec png and err png
                    self.write_images(np.abs(x_complex_concat)*255 ,os.path.join(self.args.image_folder,dataset_name+'_img_Rec'+'.png'))
                    self.write_images(np.abs(err_TSE_concat)*255 ,os.path.join(self.args.image_folder,dataset_name+'_err_Rec'+'.png'))
                    # save rec mat and err mat
                    savemat(os.path.join(self.args.image_folder,dataset_name+'_err_img'),{'img':err_TSE_concat})
                    savemat(os.path.join(self.args.image_folder,dataset_name+'_rec_img'),{'img':x_complex_concat})
                
                if min_rmse > rmse :
                    result_all[0,1] = rmse
                    min_rmse = rmse
                
                if max_snr < snr :
                    result_all[0,2] = snr
                    max_snr = snr
                # save evaluation indicators
                self.write_Data(result_all,1)
    # save image
    def write_images(self, x,image_save_path):
        x = np.array(x,dtype=np.uint8)
        cv2.imwrite(image_save_path, x)
        
    # save evaluation indicators 
    def write_Data(self, result_all,i):
        with open(os.path.join(self.args.image_folder,"evaluation indicators.txt"),"w+") as f:
            for i in range(len(result_all)):
                f.writelines('current image {} PSNR : '.format(i) + str(result_all[i,0]) + \
                "    RMSE : " + str(result_all[i,1]) + "    SNR : " + str(result_all[i,2]))
                f.write('\n')

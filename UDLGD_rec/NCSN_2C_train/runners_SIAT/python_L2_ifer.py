import numpy as np
import math
from scipy.io import loadmat,savemat
import  matplotlib.pyplot as plt
def L2_image_from_edges_rect(Xhatp,DvX,DhX,beta):
	N = Xhatp.shape
	W = np.zeros(N)
	W[~(Xhatp==0)] = 1
	data_list = np.meshgrid(range(N[1]),range(N[0]))
	k2,k1 = data_list
	pi = math.pi
	FDV = (1 - np.exp(-2*pi*1j*k1/N[0]))
	FDH = (1 - np.exp(-2*pi*1j*k2/N[1]))
	EH = np.zeros(N)
	EH[0,0] =1
	X = np.fft.ifft2(((np.conjugate(FDV)*np.fft.fft2(DvX) + np.conjugate(FDH)*np.fft.fft2(DhX))/((1+beta)*(abs(FDV)**2 + abs(FDH)**2 +EH)))*(1-W) + Xhatp)
	
	return X
	
def L2_image_from_edges_rect_reverse(Xhatp):
	N = Xhatp.shape
	X = np.zeros((N[0],N[1],4)).astype(np.complex64)
	real_data,imag_data = np.real(Xhatp),np.imag(Xhatp)
	data_list = np.meshgrid(range(N[1]),range(N[0]))
	k2,k1 = data_list
	pi = math.pi
	fdx = (1 - np.exp(-2*pi*1j*k1/N[0]))
	fdy = (1 - np.exp(-2*pi*1j*k2/N[1]))
	X[:,:,0] = np.fft.ifft2(np.fft.fft2(real_data)*fdx)
	X[:,:,1] = np.fft.ifft2(np.fft.fft2(imag_data)*fdx)
	X[:,:,2] = np.fft.ifft2(np.fft.fft2(real_data)*fdy)
	X[:,:,3] = np.fft.ifft2(np.fft.fft2(imag_data)*fdy)

	return np.real(X)

def mosaic( imgs, row_num, col_num, fig_num, title_str, disp_range ):
	imgs = abs(imgs)
	H,W,C =imgs.shape 
	show_ = np.zeros((H*row_num,W*col_num))
	for i in range(row_num):
		S = imgs[:,:,col_num*i+1]
		for k in range(col_num):
			S = np.concatenate((S,imgs[:,:,col_num*i+k]),1)
		if i ==0:
			show_ = S
		else :
			show_ = np.concatenate((show_,S),0)

	plt.figure(fig_num)
	plt.imshow(show_,cmap = 'gray')
	plt.show()	
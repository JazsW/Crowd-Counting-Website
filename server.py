from flask import Flask, render_template, request
import random
import numpy as np
import cv2
import re
import base64


import os
import numpy as np 
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torch.optim as optim
from time import time
import math
import pandas as pd
import csv

import scipy.io as sio#use to import mat as dic,data is ndarray
from skimage import io, transform#

#from load_data_V2 import myDataset, ToTensor
from Network.SDCNet import SDCNet_VGG16_classify

app = Flask(__name__)

@app.route('/')
def index():
		
	mat = sio.loadmat("Test_Data/SH_partA_Density_map/rgbstate.mat")
	#print(mat)
	global rgb
	rgb = mat['rgbMean'].reshape(1,1,3) #rgbMean is computed after norm to [0-1]
	
	max_num_list = {0:22,1:7}
	# set label_indice
	label_indice = np.arange(0.5,max_num_list[0]+0.5/2,0.5)
	add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
	label_indice = np.concatenate( (add,label_indice) )

	# init networks
	label_indice = torch.Tensor(label_indice)
	class_num = len(label_indice)+1
	#div_times = 2
	
	global net
	net = SDCNet_VGG16_classify(class_num,label_indice,psize=64, pstride =64 ,div_times=2,load_weights=True).cuda()

	# test the exist trained model
	mod_path="model/SHA/best_epoch.pth"
	if os.path.exists(mod_path):
		all_state_dict = torch.load(mod_path)
		net.load_state_dict(all_state_dict['net_state_dict'])
		return render_template('index.html')
	else:
		return render_template('fail.html')



@app.route('/get_count', methods=['POST'])
def get_count():	
	
	data = request.form['imageBase64']
	if isinstance(data, str):
		print("")
	else:
		return("0")
	data3 = data.split(',')
	image = np.fromstring(data3[1], np.uint8)
	
	
	#image = io.imread("Test_Data/SH_partA_Density_map/test/images/IMG_5.jpg") #load as numpy ndarray
	image = image/255. -rgb
	
	image = image.transpose((2, 0, 1))
	to_image = torch.from_numpy(image)
	
	
	h,w = to_image.size()[-2:]
	
	DIV = 64
	ph,pw = (DIV-h%DIV),(DIV-w%DIV)
	# print(ph,pw)

	if (ph!=DIV) or (pw!=DIV):
		tmp_pad = [pw//2,pw-pw//2,ph//2,ph-ph//2]
		# print(tmp_pad)
		to_image = F.pad(to_image,tmp_pad)
		
		
	to_image = to_image.type(torch.float32)
	to_image = to_image.cuda()
		
	features = net(to_image[None,...])
	div_res = net.resample(features)
	merge_res = net.parse_merge(div_res)
	outputs = merge_res['div'+str(net.args['div_times'])]
	del merge_res
	
	pre =  (outputs).sum()
	return(pre)
	#this is the print area
	#print('Count: %.3f' %  (pre) )



#background process happening without any refreshing
@app.route('/background_process_test', methods=['POST'])
def background_process_test():
	data = request.form['imageBase64']
	if isinstance(data, str):
		print("")
	else:
		return("0")
	data3 = data.split(',')
	nparr = np.fromstring(data3[1], np.uint8)
	
	#cv2.imwrite('/image.png',nparr)
	
	int = random.randint(0, 999)
	converted_num = str(int)
	return (converted_num)
	

if __name__ == '__main__':
  app.run(debug=True)

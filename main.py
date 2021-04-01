from flask import Flask, render_template, request
import numpy as np
import base64
import os
import torch.nn as nn
from torch import Tensor, load, from_numpy, float32, device
import scipy.io as sio#use to import mat as dic,data is ndarray
from skimage import io#

#from load_data_V2 import myDataset, ToTensor
from Network.SDCNet import SDCNet_VGG16_classify

app = Flask(__name__)

@app.route('/')
def index():
	mat = sio.loadmat("rgbstate.mat")
	global rgb
	rgb = mat['rgbMean'].reshape(1,1,3) #rgbMean is computed after norm to [0-1]
	
	max_num_list = {0:22,1:7}
	# set label_indice
	label_indice = np.arange(0.5,max_num_list[0]+0.5/2,0.5)
	add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
	label_indice = np.concatenate( (add,label_indice) )

	# init networks
	label_indice = Tensor(label_indice)
	class_num = len(label_indice)+1
	
	global net
	net = SDCNet_VGG16_classify(class_num,label_indice,psize=64, pstride =64 ,div_times=2,load_weights=True)

	# use the trained model
	mod_path="best_epoch.pth"
	if os.path.exists(mod_path):
		all_state_dict = load(mod_path, map_location=device('cpu'))
		net.load_state_dict(all_state_dict['net_state_dict'])
		return render_template('index.html')
	else:
		return render_template('fail.html')



@app.route('/get_count', methods=['POST'])
def get_count():	
	
	data = request.values['imageBase64']
	data3 = data.split(',')
	image64 = base64.b64decode(data3[1])
	x = io.imread(image64, plugin='imageio')
	x = x[:,:,:3]
	
	image = x/255. -rgb
	image = image.transpose((2, 0, 1))
	to_image = from_numpy(image)
	h,w = to_image.size()[-2:]
	
	DIV = 64
	ph,pw = (DIV-h%DIV),(DIV-w%DIV)

	if (ph!=DIV) or (pw!=DIV):
		tmp_pad = [pw//2,pw-pw//2,ph//2,ph-ph//2]
		to_image = nn.functional.pad(to_image,tmp_pad)
		
		
	to_image = to_image.type(float32)
	to_image = to_image
		
	features = net(to_image[None,...])
	div_res = net.resample(features)
	merge_res = net.parse_merge(div_res)
	outputs = merge_res['div'+str(net.args['div_times'])]
	del merge_res
	
	pre =  (outputs).sum()
	#print(int(pre.item()))
	return(str(int(pre.item())))


if __name__ == '__main__':
  app.run(debug=True)
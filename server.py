from flask import Flask, render_template, request, session
import numpy as np
import base64
import os
import torch.nn as nn
from torch import Tensor, load, from_numpy, float32, device
import scipy.io as sio#use to import mat as dic,data is ndarray
from skimage import io
import sqlite3 as sql

#from load_data_V2 import myDataset, ToTensor
from Network.SDCNet import SDCNet_VGG16_classify


app = Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'


@app.route('/')
def home():
	return(go_to_home())
	
@app.route('/logout')
def logout():
	cameraID = session['ID']
	#DISTROY Camera
	con = sql.connect("database.db")
	cur = con.cursor()
	cur.execute("DELETE FROM camera WHERE cameraID="+str(cameraID))
	con.commit()
	
	session.pop('ID', None)
	return(go_to_home())
	

@app.route('/Create_Camera',methods = ['POST'])
def Create_Camera():
	try:
		Loc_Name = request.form['Loc_Name']
		with sql.connect("database.db") as con:
			con.row_factory = sql.Row
			cur = con.cursor()
			sql_query = "SELECT locationID FROM location WHERE location='"+ Loc_Name+ "'"
			cur.execute(sql_query)
			locationID = cur.fetchone();
			if(locationID[0]==""):
				return render_template("login.html",msg = "ERROR: Failed To Find Location With Name"+Loc_Name)
			#print("INSERT INTO camera (locationID, count) VALUES ("+str(locationID[0])+", 0)")
			cur.execute("INSERT INTO camera (locationID, count) VALUES ("+str(locationID[0])+", 0)")
			con.commit()
			
			#get cameraID
			sql_query = "SELECT MAX(cameraID) FROM camera"
			cur.execute(sql_query)
			cameraID = cur.fetchone();
			x = len(cameraID)
			#print(len(cameraID))
			session['ID'] = cameraID[x-1]
			con.commit()
			return(go_to_index()) 
	except:
		con.rollback()
		return render_template("login.html",msg = "ERROR: Failed To Create Camera")
	

#@app.route('/OpenCrowdPage')
def go_to_index():
	if 'ID' in session:
		#cameraID = session['ID']
		return render_template('index.html')
	else:
		return render_template("login.html",msg = "ERROR Failed To Create Session")

	
@app.route('/home')
def go_to_home():
	con = sql.connect("database.db")
	con.row_factory = sql.Row
	cur = con.cursor()
	cur.execute("select * from location")
	rows = cur.fetchall();
	cur.execute("select * from camera")
	rows2 = cur.fetchall();
	
	return render_template('home.html', rows = rows, rows2 = rows2)
	

@app.route('/login')
def go_to_login():
	return render_template('login.html')


@app.route('/get_table_data', methods=['GET', 'POST'])
def get_table_data():
	con = sql.connect("database.db")
	con.row_factory = sql.Row
	cur = con.cursor()
	cur.execute("select * from camera")
	C_rows = cur.fetchall();
	cur.execute("select * from location")
	L_rows = cur.fetchall();
	
	total_counts = []
	for i in range(len(L_rows)):
		total_counts.append(0)
	#sum all the counts
	for i in C_rows:
		total_counts[i[1]-1] += i[2]
				
	table = "["
	for i in range(len(L_rows)):
		table = table + '{"location":"'+str(L_rows[i][1])+'", '
		table = table +  '"max_occupancy":"'+str(L_rows[i][2])+'", '
		table = table +  '"count":"'+str(total_counts[i])+'"} '
		if i != len(L_rows)-1:
			table = table + ", "
	table = table + "]"
	#print(table)
	return (table)



@app.route('/get_count', methods=['POST'])
def get_count():	
	cameraID = session['ID']
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
	
	
	con = sql.connect("database.db")
	cur = con.cursor()
	cur.execute("UPDATE camera SET count="+str(int(pre.item()))+" WHERE cameraID="+str(cameraID))
	con.commit()
		
	#print(int(pre.item()))
	return(str(int(pre.item())))


if __name__ == '__main__':
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
		app.run(debug=True)
	else:
		quit()
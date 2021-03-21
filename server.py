from flask import Flask, render_template, request
import random
import numpy as np
import cv2
import re
import base64

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html', result = "")

@app.route('/my-link/')
def my_link():

  return "14"
  
@app.route('/handle_data', methods=['POST'])
def handle_data():
    projectpath = request.form['projectFilepath']
    #return projectpath
    # your code
    # return a response
    int = random.randint(0, 999)
    converted_num = str(int)
    return render_template('index.html', text = projectpath, result = converted_num)

@app.route('/saveImage', methods=['POST'])
def saveImage():
	f = request.files['file']
	sfname = 'static/images/'+str(secure_filename(f.filename))
	f.save(sfname)


@app.route('/hook', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    image_PIL = Image.open(cStringIO.StringIO(image_b64))
    image_np = np.array(image_PIL)
    return render_template('index.html', text = projectpath, result = image_np.shape)
	

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
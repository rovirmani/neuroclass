import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template

import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from pyimagesearch.resnet import ResNet
from pyimagesearch import config
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import argparse
import shutil

from werkzeug import secure_filename

UPLOAD_FOLDER = 'malaria/testing/x'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
   <head>
  <meta charset="utf-8">
  <title>particles.js</title>
  <meta name="viewport" content="width=device-width, initial-scale=12.0, minimum-scale=14.0, maximum-scale=18.0, user-scalable=no">
  <link rel="stylesheet" media="screen" href="css/style.css"><center>NeuroClass:Please Upload a File
</head>
    <div id="particles-js"></div>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form></center>
    <nav class="navbar navbar-inverse navbar-fixed-top">
    <body style = "background: rgb(255,217,0);
background: linear-gradient(90deg, rgba(255,217,0,1) 33%, rgba(255,0,108,1) 90%); padding-top: 250px; ">
    <nav class="navbar navbar-inverse navbar-fixed-top">
      <div class="container">
        <div class="navbar-header">
        </div>
      </div>
    </nav>
    <script src="../particles.js"></script>
    <script src="js/app.js"></script>
    </body>
    '''


@app.route('/show/<filename>')
def uploaded_file(filename):
    folder = 'malaria/testing/x'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    file1 = filename
    img = cv2.imread(filename)
    
    #pred = model.predict(img)
    trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

    valAug = ImageDataGenerator(rescale=1 / 255.0)
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    testGen = valAug.flow_from_directory(
	    config.TEST_PATH,
	    class_mode="categorical",
	    target_size=(64, 64),
	    color_mode="rgb",
	    shuffle=False,
	    batch_size=1)
    model = load_model('classifier.h5')
    INIT_LR=1e-1
    opt = SGD(lr=INIT_LR, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt,
	    metrics=["accuracy"])

    totalTest = len(list(paths.list_images(config.TEST_PATH)))
    predIdxs = model.predict_generator(testGen,
	    steps=(totalTest) + 1)
    
    infected = predIdxs[1][0]
    uninfected = predIdxs[1][1]
    if infected > uninfected :
        diag1 = "infected"
    else :
        diag1 = "uninfected"
    
    return render_template('template.html', filename=filename,diag = diag1)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
if __name__ == '__main__':
    app.run(debug = True)

from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory, make_response
import os
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from keras.optimizers import SGD
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D, Input, Conv2D, UpSampling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from numpy import dot
from numpy.linalg import norm

app = Flask(__name__, static_folder='static')


app.secret_key = 'welcome'

X = []
Y = []
question = []
answer = []

dataset = pd.read_csv("Dataset.csv")
dataset = dataset.values
for i in range(len(dataset)):
    q = dataset[i,0].strip().lower()
    a = dataset[i,1].strip().lower()
    question.append(q)
    answer.append(a)
#generating question and answer into numeric vector
question_vector = TfidfVectorizer()
X = question_vector.fit_transform(question).toarray()

answer_vector = TfidfVectorizer()
Y = answer_vector.fit_transform(answer).toarray()
features = answer_vector.get_feature_names_out()
#normalizing generated vector
sc1 = StandardScaler()
X = sc1.fit_transform(X)
#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
#defining ML algorithm object
cnn_model = Sequential()
#defining dense layer with 128 neurons to filter input data 
cnn_model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
#defining dropout layer to remove irrelevant features
cnn_model.add(Dropout(0.5))
#defining another layer to further filter features
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
#defining output layer
cnn_model.add(Dense(y_train.shape[1], activation='softmax'))
#compiling training and loading model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy')
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 8, epochs = 1000, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/TextChatbot', methods=['GET', 'POST'])
def TextChatbot():
    return render_template('TextChatbot.html', msg='')

@app.route('/ChatData', methods=['GET', 'POST'])
def ChatData():
    if request.method == 'GET':
        global question, answer, X, question_vector, sc1, cnn_model
        data = request.args.get('mytext')
        arr = data.strip().lower()
        testData = question_vector.transform([arr]).toarray()
        testData = sc1.transform(testData)
        testData = testData[0]
        print(testData.shape)
        max_accuracy = 0
        index = -1
        for i in range(len(X)):
            predict_score = dot(X[i], testData)/(norm(X[i])*norm(testData))
            if predict_score > max_accuracy:
                max_accuracy = predict_score
                index = i
        output = ""
        if index != -1 and max_accuracy > 0.30:
            output = answer[index]                 
        else:
            output = "Unable to predict answers. Please Try Again"
        print(output)    
        response = make_response(output, 200)
        response.mimetype = "text/plain"
        return response    
            

@app.route('/TrainML', methods=['GET', 'POST'])
def TrainML():
    f = open('model/cnn_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    loss = data['loss']
    val_loss = data['val_loss']
    plt.figure(figsize=(6,5))
    #plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, color = 'green')
    plt.plot(val_loss, color = 'red')
    plt.legend(['ML Training Loss', 'ML Validation Loss'], loc='upper left')
    plt.title('ML Training & Validation Loss Graph')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()   
    return render_template('ViewResult.html', data="ML Training & Validation Loss Graph", img = img_b64)


if __name__ == '__main__':
    app.run(debug=True)
    

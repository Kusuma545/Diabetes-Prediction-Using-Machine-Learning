from django.shortcuts import render, redirect
from . import views
# Create your views here.
import pandas as pd
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from django.http import HttpResponse
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import os
from .models import *
from django.contrib.auth.models import User
from sklearn.metrics import accuracy_score
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

# Register function
def Registration(request):
    if request.method=='POST':
        Name = request.POST['username']
        email=request.POST['email1']
        password=request.POST['password']
        conpassword=request.POST['cpassword']
        contact=request.POST['contact']
        print(Name,email,password,conpassword,contact)
        if password==conpassword:
            dc = User.objects.filter(email=email,password=password)
            if dc:
                msg='Account already exists'
                return render(request,'Registration.html',{'msg':msg})
            else:
                user=User(username=Name,email=email,password=password)
                user.save()
                return render(request,'login.html')
                
        else:
            msg='passwords not matched '
            return render(request,'Registration.html',{'msg':msg})
    return render(request,'Registration.html')


def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']

        d=User.objects.filter(email=lemail,password=lpassword).exists()
        if d:

            return redirect('upload')
        else:
            msg='login failed'
            return render(request,'login.html',{'msg':msg})

    return render(request,'login.html')

def userhome(request):
    
    return render(request,'userhome.html')


def upload(request):
    global data, path
    if (request.method == 'POST'):
        file = request.FILES['file']
        d = dataset(data=file)
        


        fn = d.filename()
        path = 'app\static\dataset'+fn

        data = pd.read_csv('app\static\dataset\diabetes.csv')
        datas = data.iloc[:100,:]
        x = datas.to_html()

        return render(request, 'upload.html', {'table':x})
    return render(request, 'upload.html')

def splitdata(request):
    global x_train,x_test,y_train,y_test,x,y,data,path
    if request.method == "POST":
        size = request.POST['split']
        size = int(request.POST['split'])
        size = size / 100
        data=pd.read_csv('app\static\dataset\diabetes.csv')
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1]
        x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=41)
        messages.info(request,"Data Splits Succesfully")
    return render(request,'splitdata.html')

import pickle
def modeltrain(request):
    global x_train,x_test,y_train,y_test,x,y,data
    if request.method == "POST":
        model = request.POST['algo']

        if model == "1":
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            acc1 = accuracy_score(y_pred, y_test)
            acc1 = acc1*100
            acc1
            msg='Accuracy of knn  ' + str(acc1)
            return render(request,'modeltrain.html',{'msg':msg})
        
        elif model == "2":

            dtc = DecisionTreeClassifier()
            dtc.fit(x_train, y_train)
            y_pred = dtc.predict(x_test)
            acc2 = accuracy_score(y_pred, y_test)
            acc2 = acc2*100
            acc2
            msg='Accuracy of Decision tree : ' + str(acc2)
            return render(request,'modeltrain.html',{'msg':msg})
        
        elif model == "3":
            lr=LogisticRegression()
            lr.fit(x_train,y_train)
            y_pred=lr.predict(x_test)
            acc8=accuracy_score(y_pred,y_test)
            acc8=acc8*100
            acc8
            msg='Accuracy of LogisticRegression : ' + str(acc8)
            return render(request,'modeltrain.html',{'msg':msg})

        elif model == "5":
            svc=SVC()
            svc.fit(x_train,y_train)
            y_pred=svc.predict(x_test)
            acc5=accuracy_score(y_pred,y_test)
            acc5=acc5*100
            acc5
            msg='Accuracy of random forest : ' + str(acc5)
            return render(request,'modeltrain.html',{'msg':msg})
        
        elif model == "4":

            model=Sequential()
            model.add(Dense(1000,input_dim=8,activation='relu'))
            model.add(Dense(500,activation='relu'))
            model.add(Dense(300,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1,activation='softmax'))
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            model.summary()
            model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=20,epochs=20,verbose=1)
            prediction=model.predict(x_test)
            acc4 = accuracy_score(prediction,y_test)
            acc4=acc4*100
            acc4
            msg='Accuracy of AdaBoostClassifier : ' + str(acc4)
            return render(request,'modeltrain.html',{'msg':msg})
       
        elif model == "6":
            hybrid = list()
            hybrid.append(('RF', RandomForestClassifier()))
            hybrid.append(('NB', GaussianNB()))
            model_h=StackingClassifier(estimators=hybrid)
            model_h.fit(x_train, y_train)
            y = model_h.predict(x_test)
            acc6 = accuracy_score(y_test, y)
            acc6=acc6*100
            acc6
            msg='Accuracy of hybrid : ' + str(acc6)
        return render(request,'modeltrain.html',{'msg':msg})
    return render(request,'modeltrain.html')

def prediction(request):
    global x_train,x_test,y_train,y_test,x,y
    

    if request.method == 'POST':

        a = int(request.POST['f1'])
        b = int(request.POST['f2'])
        c = int(request.POST['f3'])
        d = int(request.POST['f4'])
        e = int(request.POST['f5'])
        f = int(request.POST['f6'])
        g = int(request.POST['f7'])
        h = int(request.POST['f8'])
        
       
        PRED = [[a,b,c,d,e,f,g,h]]
       
        lr = LogisticRegression()
        lr.fit(x_train,y_train)
        y_pred = np.array(lr.predict(PRED))

        if y_pred ==0:
            msg = '''The prediction result is the person is having diabetes,|
            DOCTORS: If you have been diagnosed with a hormonal condition such as diabetes or thyroid, you might be advised by your doctor to consult an endocrinologist.|
            MEDICINES:  dulaglutide (Trulicity)|
                        exenatide (Byetta)|
                        exenatide extended-release (Bydureon BCise)|
                        liraglutide (Saxenda, Victoza)|
                        lixisenatide (Adylyxin)|
                        semaglutide (Ozempic)|
                        tirzepatide (Mounjaro),|
            HOW TO CONTROL DIABEYTES: Lose extra weight. Losing weight reduces the risk of diabetes. ...|
            Be more physically active. There are many benefits to regular physical activity. ...|
            Eat healthy plant foods. Plants provide vitamins, minerals and carbohydrates in your diet. ...|
            Eat healthy fats. ...|
            Skip fad diets and make healthier choices'''
        else:
            msg = 'The prediction result is the person is not having diabetes'

        msg = msg.split('|')
        print(msg)


        return render(request,'prediction.html',{'msg':msg})

    return render(request,"prediction.html")


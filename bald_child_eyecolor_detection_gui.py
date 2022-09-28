#Import necessary libraries

from cProfile import label
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from turtle import heading
from PIL import Image,ImageTk
import numpy as np
import cv2
from  mtcnn.mtcnn import MTCNN
import os
from keras.models import load_model
model = load_model("../Downloads/Resources/bald_detection.h5")
model2 = load_model("../Downloads/Resources/Age_Sex_Detection.h5")

#Initialization of GUI
top = tk.Tk()
top.geometry('500x500') 
top.title('Bald Detection')
top.configure(background='#CDCDBD')

#Initialization the labels
label1 = Label(top,background="#CDCDBD",font=('arial',14))
label2 = Label(top,background="#CDCDBD",font=('arial',14))
sign_image = Label(top)

color_name = ("Blue","Bluish Gray","Brown","Brownish Gray","Brownish Black","Green","Greenish Gray","Others") 
eyeColor =  {    
    color_name[0] : ((166, 21, 50), (240, 100, 85)),
    color_name[1] : ((166, 2, 25), (300, 20, 75)),
    color_name[2] : ((2, 20, 20), (40, 100, 60)),
    color_name[3] : ((20, 3, 30), (65, 60, 60)),
    color_name[4] : ((0, 10, 5), (40, 40, 25)),
    color_name[5] : ((60, 21, 50), (165, 100, 85)),
    color_name[6] : ((60, 2, 25), (165, 20, 65))
}

#compare the eye color with above defined colours to get the specific eye color
def check_color(hsv, color):
    
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and (hsv[1] <= color[1][1]) and \
    (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False

#define eye color category rules in HSV space
def find_class(hsv):
    color_id = 7
    for i in range(len(color_name)-1):
        if check_color(hsv, eyeColor[color_name[i]]) == True:
            color_id = i
        
    return color_id

def get_eye_color(file_path):
    detector =  MTCNN()  
    image1 = cv2.imread(file_path,cv2.IMREAD_COLOR)
    result = detector.detect_faces(image1)
    imgHSV = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    h, w = image1.shape[0:2]
    imgMask = np.zeros((image1.shape[0], image1.shape[1], 1))

    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']
    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15 # approximate
   
    cv2.circle(imgMask, left_eye, int(eye_radius), (255,255,255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255,255,255), -1)

    cv2.circle(image1, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image1, right_eye, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(color_name), np.float)

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y,x])] += 1 

    color_index = np.argmax(eye_class[:len(eye_class)-1])
    return color_index

# To determine whether image uploaded is child or not.
def get_age(filepath):
    global label_packed
    image_age = Image.open(filepath)
    image_age = image_age.resize((80,80))
    image_age = np.expand_dims(image_age,axis=0)
    image_age = np.array(image_age)
    image_age = np.delete(image_age,0,1)
    image_age = np.resize(image_age,(80,80,3))
    print(image_age.shape)                          # to debugg the code
    image_age = np.array([image_age])/255
    pred= model2.predict(image_age)
    age = int(np.round(pred[1][0]))
    return age

#Defining Detect function which detects the baldness of the person in the image using the model defined earlier. 
def Detect(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((80,80))
    image = np.expand_dims(image,axis=0)
    image = np.array(image)
    image = np.delete(image,0,1)
    image = np.resize(image,(80,80,3))
    print(image.shape)                  # to debugg the code
    detect = ['Not Bald','Bald'] 
    image = np.array([image])/255
    pred_bald = model.predict(image)
    baldornot = int(np.round(pred_bald))
    i = get_eye_color(file_path)
    age_child = get_age(file_path)

    #print("Bald Prediction "+ detect[baldornot])
    #print(str(i))
    #print("Dominant Eye Color: "+ color_name[i])
    #print("Age of the Person in Image: "+ str(age_child)) 
    
    if (age_child>5):
        label1.configure(foreground="#011638",text=detect[baldornot])
    else:
        label1.configure(foreground="#011638",text="Child")
    if(i==None):
       label2.configure(foreground="#011638",text="Some error occured")
    else:
       label2.configure(foreground="#011638",text="Eye Color: "+color_name[i])
   
    
# Detect Button defination
def show_Detect_Btn(file_path):
    Detect_b=Button(top,text="Detect Image",command=lambda: Detect(file_path),padx=10,pady=5)
    Detect_b.configure(background="#368056",foreground="white",font=('arial',10,'bold'))
    Detect_b.place(relx=0.38,rely=0.90)

#Define Upload Image Function
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        label2.configure(text='')
        show_Detect_Btn(file_path)
    except:
        pass

upload= Button(top,text="Upload an Image",command=upload_image,padx=10,pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
upload.pack(side="bottom",pady=50)
sign_image.pack(side="bottom",expand=True)

label1.pack(side="bottom",expand=True)
label2.pack(side="bottom",expand=True)
heading = Label(top,text="Bald/Child & Eye Color Detector",pady=10,font=('Monospaced',20,"bold"))
heading.configure(background="#CDCDBD",foreground="#000011")
heading.pack()
top.mainloop()
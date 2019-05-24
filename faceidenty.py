import dlib
import numpy as np
import cv2
import os
import json

detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')                 #使用CNN进行人脸检测的检测算子
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')                          #获取人脸区域中的五官几何点区域
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')       #获取ResNet模型

imagePath = './photo/'
data = np.zeros((1,128))
label = []

for file in os.listdir(imagePath):
    if '.jpg' in file or '.png' in file or '.jpeg' in file:
        fileName = file
        labelName = file.split('_')[0]
        print('current image: ',file)
        print('current label: ',labelName)

        img = cv2.imread(imagePath + file)
        if img.shape[0]*img.shape[1] > 500000:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        dets = detector(img, 1)                                     #使用检测算子检测人脸，返回所有检测到的人脸区域
        for k, d in enumerate(dets):
            rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(),d.rect.bottom())
            shape = sp(img, rec)                                    #获取landmark
            face_descriptor = facerec.compute_face_descriptor(img, shape)       #使用resNet获取128维的人脸特征向量
            faceArray = np.array(face_descriptor).reshape((1,128))
            data = np.concatenate((data, faceArray))
            label.append(labelName)
            cv2.rectangle(img, (rec.left(), rec.top()),(rec.right(), rec.bottom()), (0, 255, 0), 2)
        cv2.waitKey(2)
        cv2.imshow('image',img)

data = data[1:, :]
np.savetxt('faceData.txt',data,fmt='%f')

labelFile=open('label.txt', 'w')
json.dump(label, labelFile)
labelFile.close()

cv2.destroyAllWindows()

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'ImagesAttendance'                                                         #image directory is taken
images = []                                                                       #list of images to import to store in the list
classNames = []                                                                   #to take the names directly from the image file name
myList = os.listdir(path)                                                         #to grab the list of images in to the folder
print(myList)                                                                     #to print the list of names
for cl in myList:                                                                 #to import the images one by one
    curImg = cv2.imread(f'{path}/{cl}')                                           #to read the image with file path
    images.append(curImg)                                                         # appending images from the list to path
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                 #to convert the image to RGB code
        encode = face_recognition.face_encodings(img)[0]                           #to encode the recognized face
        encodeList.append(encode)                                                  #to append the encoded list
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:                                        #to open the attendance list
        myDataList = f.readlines()                                                 #to read the lines in the data list
        nameList = []                                                              #empty list
        for line in myDataList:
            entry = line.split(',')                                                #to split the lines
            nameList.append(entry[0])                                              #to append the namelist with entry 0
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')                                    #to print the check in time
            f.writelines(f'{name},{dtString}\n')                                   #to print the name of the person in the screen

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    cv2.imshow('webcam',img)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)                                #to implement the square size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)                                    #to implement the colour code for the square

    facesCurFrame = face_recognition.face_locations(imgS)                           #to locate the name list
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            #break
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
#markAttendance(name)
#cv2.destroyAllWindows()
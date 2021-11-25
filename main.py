import face_recognition
import numpy
import dlib
import cv2.cv2 as cv2
img1_raw = face_recognition.load_image_file('/Users/vedantgupta/PycharmProjects/pythonProject1/venv/imgs/IMG_1611.jpg')
img1_raw = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB)
img2_raw = face_recognition.load_image_file('/Users/vedantgupta/PycharmProjects/pythonProject1/venv/imgs/IMG_1610.jpg')
img2_raw = cv2.cvtColor(img2_raw, cv2.COLOR_BGR2RGB)

img1_raw=cv2.rotate(img1_raw, cv2.ROTATE_90_CLOCKWISE)
img2_raw=cv2.rotate(img2_raw, cv2.ROTATE_90_CLOCKWISE)

faceLoc = face_recognition.face_locations(img1_raw)[0]
encodeimg1 = face_recognition.face_encodings(img1_raw)[0]
cv2.rectangle(img1_raw, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(img2_raw)[0]
encodeimg2 = face_recognition.face_encodings(img2_raw)[0]
cv2.rectangle(img2_raw, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results=face_recognition.compare_faces([encodeimg1], encodeimg2)
print(results)

cv2.imshow('Face1', img1_raw)
cv2.imshow('Face2', img2_raw)
cv2.waitKey(0)

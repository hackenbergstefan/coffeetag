import pickle
import sys

import cv2
import face_recognition

try:
    data = pickle.loads(open('face_encodings.pickle', 'rb').read())
except:
    data = {'encodings': [], 'names': []}

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set Width
cap.set(4, 480) # set Height
while True:
    ret, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')

    for (y, xw, yh, x) in boxes:
        cv2.rectangle(img, (x, y), (xw, yh), (255, 0, 0), 2)

    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elif k == 32: # 'SPACE':
        name = sys.argv[1]
        print('save', name)
        for encoding in face_recognition.face_encodings(rgb, boxes):
            data['encodings'].append(encoding)
            data['names'].append(name)
        with open('face_encodings.pickle', 'wb') as fout:
            fout.write(pickle.dumps(data))
        break

cap.release()
cv2.destroyAllWindows()

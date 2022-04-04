import cv2 as cv

fd = cv.CascadeClassifier('../machineLearning/base/tranning/config/haarcascade_frontalface_default.xml')
ed = cv.CascadeClassifier('../machineLearning/base/tranning/config/haarcascade_eye.xml')

while True:
    images = cv.VideoCapture(0)
    image = images.read()[1]
    faces = fd.detectMultiScale(image, 1.3, 5)
    for l,t,w,h in faces:
        a,b = int(w/2), int(h/2)
        cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360, (255, 0, 245), 2)
        face = image[t:t + h, l:l + w]
        eyes = ed.detectMultiScale(face, 1.3, 5)
        for l,t,w,h in eyes:
            f,g = int(w/2), int(h/2)
            cv.ellipse(image, (l + f, t + g), (f, g), 0, 0, 360, (255, 0, 255), 2)    
    cv.putText(image, '人脸跟踪'.format(), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 2,
                   (255, 255, 255), 6)
    cv.imshow('人脸跟踪',image)
    if cv.waitKey(33) == 27:
        break
cv.release()
cv.destoryAllwindows()
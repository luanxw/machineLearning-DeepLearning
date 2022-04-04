import cv2 as cv 

# video_capture = cv.VideoCapture(0)
# frame = video_capture.read()[1]
# cv.imshow('videoCapture',frame)
# cv.waitKey()
# video_capture.release()
# cv.destoryAllwindows()
def search_flies(directory):
    '''
    检索 directory 目录下所有的jpg文件返回字典目录
    '''
    objects = {}
    for curdir ,subdir , files in  os.walk(directory):
        for file in files:
            if file.endswith('jpg'):
                label = curdir.split(os.path.sep)[-1]
                if label not in objects:
                    objects[label] = []
                url = os.path.join(curdir, file)
                objects[label].append(url)
    return objects
data = search_flies('../machineLearning/data/computer')

while True:
    images = cv.VideoCapture(0)
    image = images.read()[1]
    cv.imshow('摄像头捕捉截图',image)
    if cv.waitKey(33) == 27:
        break
cv.release()
cv.destoryAllwindows()
fd = cv.CascadeClassifier('../machineLearning/base/training/config/haarcascade_frontalcatface.xml')
ed = cv.CascadeClassifier('../machineLearning/base/training/config/haarcascade_eye.xml')


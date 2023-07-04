import cv2 as cv
import matplotlib.pyplot as plt

config_file="C:\\Users\\91736\\OneDrive\\Desktop\\models\\model-4\\ssd_inception_v2_coco_2017_11_17.pbtxt"
frozen_model="C:\\Users\\91736\\OneDrive\\Desktop\\models\\model-4\\frozen_inference_graph.pb"

model=cv.dnn_DetectionModel(frozen_model,config_file)

classLables=[]
file_name="C:\\Users\91736\\OneDrive\\Desktop\\models\\labels.txt"
with open(file_name,'rt') as fpt:
    classLables=fpt.read().rstrip('\n').split('\n')
    #classLables.append(fpt.read())

print(classLables)

print(len(classLables))

model.setInputSize(320,320)
model.setInputScale(1.0/127.5) #255/2=127.5
model.setInputMean((127.5,127.5,127.5)) #mobilenet=>[-1,1]
model.setInputSwapRB(True)

img=cv.imread("C:\\Users\\91736\\OneDrive\\Desktop\\models\\test imgs\\7.jpg")

#img_resize=cv.resize(img,(600,400))

#cv.imshow('IMG',img) #BGR

#RGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)

#cv.imshow('RGB',RGB)

ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.06)
print(ClassIndex)

font_scale=3
font=cv.FONT_HERSHEY_PLAIN

if (len(bbox)>0) and (len(ClassIndex)>0) and (len(confidence)>0):
    for ClassInd, conf, boxes in zip (ClassIndex.flatten(), confidence.flatten(), bbox):
        cv.rectangle(img,boxes,(0,0,255),2)
        cv.putText(img,classLables[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
        # cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), 2)
        # cv2.putText(img, text, (text_offset_x,text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

#demo=cv.cvtColor(img, cv.COLOR_BGR2RGB)
img=cv.resize(img,(800,600))
cv.imshow('Demo',img)

cv.waitKey(0)

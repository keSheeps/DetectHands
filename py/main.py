import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

#constants
#640x480 始点と終点
startx=0
starty=280
endx=640
endy=480
#入力を認識する白の割合(%)
threshold_ratio=0.5 
#色相の始点/終点(0~179)
hue_min=0
hue_max=20
#彩度の下限/上限(0~255)
sat_min=40
sat_max=255
#明度の下限/上限(0~255)
val_min=0
val_max=255
#waitkeyの時間
wait=1
#カメラの番号
camera_id=0

#calculations
linesWidth = (int)(endx - startx)
lineWidth = (int)(linesWidth / 16)
lineHeight = (int)(endy - starty)
lineArea = (int)(lineWidth * lineHeight)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        hue = cv2.inRange(hue,hue_min,hue_max)
        sat = cv2.inRange(sat,sat_min,sat_max)
        val = cv2.inRange(val,val_min,val_max)

        for line in range(0,16):
            count=0
            for x in range(startx+lineWidth*line,startx+lineWidth*(line+1)):
                for y in range(starty,endy):
                    if(hue[y][x] and sat[y][x] and val[y][x]):
                        count+=1
            if((float)(count/lineArea)>threshold_ratio):
                print(line,end=",")
        print("-1",flush=True)
        cv2.imshow('hue', hue)
        cv2.imshow('saturation', sat)
        cv2.imshow('value', val)

        if cv2.waitKey(wait) & 0xFF == ord('q'):
            print(hue)#[たて][よこ]
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

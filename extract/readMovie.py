# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

startPt = (0, 0)
endPt   = (0, 0)
preview = None
# ドラッグによるトリミング
def mouseEvent1(event, x, y, flags, param):
    global startPt
    global endPt
    if event == cv2.EVENT_LBUTTONDOWN:
        startPt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        endPt = (x, y)
        
    if flags & cv2.EVENT_FLAG_LBUTTON:
        preview[:] = frame[:]
        cv2.rectangle(preview, startPt, (x, y), (0, 0, 255))

# サイズ指定によるトリミング
# 切り取りサイズの指定
RATO = 10
TRIM_WIDTH  = 100 * RATO
TRIM_HEIGHT = 100 * RATO
def mouseEvent2(event, x, y, flags, param):
    global startPt
    global endPt
    if event == cv2.EVENT_LBUTTONUP:
        startPt = (x, y)
        endPt   = (x + TRIM_WIDTH, y + TRIM_HEIGHT)
        preview[:] = frame[:]
        cv2.rectangle(preview, startPt, endPt, (0, 0, 255))

# コールバック関数を指定する（mouseEvent1 / 2）
# 1でやるといろいろ問題になる
CALLBACK = mouseEvent2

cap = cv2.VideoCapture('movie/2016_08_25_side.mp4')
width    = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height   = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
fps      = cap.get(cv2.cv.CV_CAP_PROP_FPS)
frameNum = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print 'width :', width
print 'height:', height
print 'fps   :', fps
print 'frame num:', frameNum
endPt = (int(width), int(height))

cv2.namedWindow('select', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('select', CALLBACK)
ret, frame = cap.read()
preview = frame.copy()
while(ret):
    cv2.imshow('select', preview)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        ret, frame = cap.read()
        preview[:] = frame[:]

if not os.path.exists('extract'):
    os.mkdir('extract')

print startPt, endPt
trimWidth  = endPt[0] - startPt[0]
trimHeight = endPt[1] - startPt[1]
out = cv2.VideoWriter('extract/output.avi', 0, int(round(fps)), (trimWidth / RATO, trimHeight / RATO))

i = 0
while(ret):
    trim = frame[startPt[1]:endPt[1], startPt[0]:endPt[0]]
    trim = cv2.resize(trim, None, fx=1.0/RATO, fy=1.0/RATO)
    out.write(trim)
    gray = cv2.cvtColor(trim, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', gray)

    cv2.imwrite('extract/img{}.png'.format(i), gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    i += 1


cap.release()
out.release()
cv2.destroyAllWindows()

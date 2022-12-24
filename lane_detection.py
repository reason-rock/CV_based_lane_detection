# -*- coding: utf-8 -*-
import cv2
from matplotlib.patches import Polygon
from cv2 import Canny #opencv import
import numpy as np #import numpy
import matplotlib.pyplot as plt #ROI 지정
import os
cap = cv2.VideoCapture("test.mp4")


def pre_processing(image): # 캐니 함수 지정
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Opencv로 rgb 파일 흑백으로 변경
    blur = cv2.GaussianBlur(gray, (5,5), 0) #가우시안 블러 적용 ((커널,커널),편차) / 캐니에 5*5 커널 있어서 선택적으로 적용
    pre_processing = cv2.Canny(blur, 50, 150) #블러 이미지에 캐니 적용 low/high threshhold 50,150
    return pre_processing

def roi_set(image):
    roi = np.array([[(583,873), (1540, 890),(1243, 616), (979,616)]])
    zeros = np.zeros_like(image)
    cv2.fillPoly(zeros,roi,255) #roi 영역 흰색으로 마스킹
    roi_set = cv2.bitwise_and(image, zeros) #bitwise 연산
    return roi_set

def make_coordinates(image, xy_value): #좌표 지정
    lane_gradient, y_intecept = xy_value
    y1 = 890 #height start from bottom
    y2 = int(y1*(4/5)) #높이 대비 라인이 끝나는 점 위치 지정
    #print(image.shape) #세로, 가로, 채널수
    x1 = int((y1 - y_intecept)/lane_gradient) #좌표 지정
    x2 = int((y2 - y_intecept)/lane_gradient)
    return np.array([x1, y1, x2, y2])

def draw_line(image, hough_lines): #검정 배경 만들고 위에 차선 그음
    line_layer = np.zeros_like(image) #빈 어레이 만듦
    if hough_lines is not None: #라인 값이 들어올때
        for x1, y1, x2, y2 in hough_lines:
            cv2.line(line_layer, (x1, y1), (x2, y2), (0,255,0), 5) #검은 화면에 좌표1, 좌표2, 선색깔, 선두께로 라인 그음
    return line_layer

def lane_define(image, hough_lines):
    left_raw = [] 
    right_raw = [] #왼/오른쪽 차선 값 임시로 받아오는 어레이
    for lane in hough_lines:
        x1, y1, x2, y2 = lane.reshape(4) #2차원 배열을 1차원으로 변환
        xy_value = np.polyfit((x1,x2),(y1,y2), 1 )
        #print(xy_value)#print [lane_gradient,  Y intecept]
        lane_gradient = xy_value[0] #파라미터 0에 기울기 저장
        y_intecept = xy_value[1] #파라미터 1에 인터셉트 값 저장
        #print(format(lane_gradient, 'f'), "lane_gradient")
        #print("----")
        #print(y_intecept, "intercep")
        if lane_gradient < 0: # 기울기 기준으로 좌/우 라인 구분
            left_raw.append((lane_gradient, y_intecept))
        else:
            right_raw.append((lane_gradient, y_intecept))

  

    #print(left_raw, 'left') #왼쪽 값
    #print(right_raw, 'right') #오른쪽 값
    left_average = np.average(left_raw, axis=0)
    right_average = np.average(right_raw, axis=0)
    #print(left_average, 'left lane_gradient') #기울기 평균값
    #print(right_average, 'right lane_gradient')
    left_line = make_coordinates(image, left_average)
    right_line = make_coordinates(image, right_average)
    return np.array([left_line, right_line])

while(cap.isOpened()):
    _, frame = cap.read() #프레임 불러오기
    pre_processed_img = pre_processing(frame) #전처리
    roi_inage = roi_set(pre_processed_img) #ROI 설정
    cv2.imshow("ROI", roi_inage)
    cv2.imshow("RAW", frame)
    hough_lines = cv2.HoughLinesP(roi_inage, 1 , np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    try:
        defined_lines = lane_define(roi_inage, hough_lines) #라인 도출
    except:
        pass
    try:
        line_output= draw_line(frame, defined_lines) #라인 시각화
    except:
        pass
    try:
        final_image = cv2.addWeighted(frame, 0.8, line_output, 1, 1) #이미지 두개 합침 / 이미지1, 투명도, 이미ㅈ2, 투명도, rgb 픽셀 밝기
    except:
        pass
    try:
        cv2.imshow("Lane_detection", final_image)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# Realtime-BI(Bio Information)-Predict-Program


![Realtime_BI_Predict_Program](https://user-images.githubusercontent.com/44527599/120924362-68ffdb80-c70e-11eb-8089-a8b64cf23ba1.png)

본 프로그램은 웹 카메라로 사람의 얼굴 영상을 실시간으로 찍어 해당 영상을 통해 심박수, 호흡수, 스트레스 지수를 예측하는 프로그램입니다.
특히 딥러닝을 이용하여 rPPG 및 remote Respiration 신호를 동시에 예측하고, 이를 실시간 그래프로 보여줍니다. 



#### 기본 설정 ####
시스템 환경: Window 10
사용 프로그램: Anaconda(Spyder)
파이썬 버전: 3.7

1. 먼저 새 아나콘다에서 새로운 가상환경을 만듭니다.
2. conda install -c conda-forge dlib 명령어를 이용해 dlib을 먼저 설치합니다.
3. 올려둔 requirments.txt을 이용하여 패키지를 설치해줍니다. (pip install –r requirements.txt --user)
4. 만약 로지텍 웹캠을 이용하실 경우, 프로그램 실행 전 Logitech capture를 실행해주시기 바랍니다.
5. 만약 외부 카메라를 사용하실 경우, RealTime_BI_Predict_Program.py의 68 줄의 
   cap = cv2.VideoCapture(카메라 번호)를 설정해주시길 바랍니다.
6. dlib의 face detector를 사용하기 위해 실행 프로그램의 같은 폴더에 shape_predictor_68_face_landmarks.dat을 넣어주시기 바랍니다.
(reference site: https://github.com/davisking/dlib-models)



* It is just for research purpose, and commercial use is not allowed!!

import numpy as np
import cv2
from multiprocessing import Process, Queue, Manager, Value

import time
import dlib


import os

from scipy.signal import find_peaks
import heartpy as hp
import matplotlib.pyplot as plt
from scipy import signal, integrate
from numpy import log as ln
import math

import keyboard as kb
from pynput import keyboard 

import threading
import sys
import psutil


from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGraphicsOpacityEffect, QVBoxLayout, QHBoxLayout
from PyQt5 import QtGui

from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot, QByteArray , Qt
from PyQt5.QtGui import QImage, QIcon, QMovie
from PyQt5 import uic


import multiprocessing as mp
import datetime

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


form_class = uic.loadUiType("./RealTime_Program.ui")[0]
form_widget = uic.loadUiType("./load.ui")[0]

class RealTime_PPG_Predict(object):
    def __init__(self, ):
        self.q_frame = Queue(maxsize = 2400)  #Queue max size 설정 X -> 실시간으로 돌아가서
        self.q_cheek = Queue(maxsize = 2400)
        self.q_forehead = Queue(maxsize = 2400)
        self.q_interface = Queue(maxsize = 2400)
        #q_bpm = Queue(maxsize = 100)
        #q_rr = Queue(maxsize = 100)
        self.q_face_detector = Queue(maxsize = 100)

        self.detect_face = Value('i',1)
        
        self.current_bpm = 0
        self.current_rr = 0
        self.current_stress = 0


########################## Get_Image #########################
    def webcam_get_image(self,):
        #multiprocess로 실행을 할 모듈 150 frame ARRAY
        
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        prev_time = 0
        interval = 0.033

        while(1):
            ret, frame = cap.read()

            if ret == True:
                self.q_frame.put(frame)
                self.q_interface.put(frame)
                    

        cap.release()
        
    
    
    def ROI_image(self, ):
        #소비자
        #queue에서 image를 get하여 ROI를 구한 후, queue에 저장.
        #150 프레임에 한 번 얼굴인식해서 프레임 자르기

    
        detector = dlib.get_frontal_face_detector() 
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        ALL = list(range(0,68))
        index = ALL
        

        i = 0
        top = 0
        left =0
        right = 0 
        center_y = 0 
        forehead_line = 0 
        cheek_line = 0
        
        while(1):
            if(self.q_frame.empty()):

                time.sleep(0.033)              
                
            else:
                #dlib
                frame = self.q_frame.get()
               
                
                if(i%30 == 0): #1초마다 ROI 업데이트
                    i = 0

                    dets = detector(frame, 1)
                    
                    self.q_face_detector.put(dets)

                    list_points = []
                    
                    
                    for face in dets:
                        shape = predictor(frame, face)
                        
                        for p in shape.parts():
                            list_points.append([p.x, p.y])
                        
                        list_points = np.array(list_points)
                    
                    

                    if(len(list_points)==0):
                        self.detect_face.value = 0   # can't detect face
                        f = open("error.txt","w")
                        f.write("can't detect face!!\n")
                        f.close()
                        # q_frame 다 비워버리기
                        
                        while not self.q_frame.empty():
                            self.q_frame.get()
                        while not self.q_cheek.empty():
                            self.q_cheek.get()
                        while not self.q_forehead.empty():
                            self.q_forehead.get()
                            
                            
                        
                        
                        continue
                    else:
                        self.detect_face.value = 1   # detect face

                    center_x = list_points[30][0]  #31 center of nose
                    center_y = list_points[30][1]
                    
                    #resize
                    width = 140
                    height = 120
                    
                    top = int(center_y - height/2)
                    bottom = int(center_y + height/2)
                    right = int(center_x + width/2)
                    left = int(center_x - width/2)
                    
                   
                    cheek_line = int(center_y + height/3)
                    forehead_line = int(top + height/3)
                
                B,G,R = cv2.split(frame) 
                frame = cv2.merge([R,G,B])
                
                cheek = frame[center_y:cheek_line, left:right]
                forehead = frame[top:forehead_line, left:right]

                self.q_cheek.put(cheek)
                self.q_forehead.put(forehead)
             
                i = i + 1
            
 
            
       
    

    def get_150_frame(self, image_cheek, image_forehead ):   # manager.list로 image_cheek랑 image_forehead 저장?? 
        
        #queue -> list, 600개의 프레임을 리스트로 저장하여 반환하기. 
        i=0

        while i <150:
            if self.q_cheek.qsize() * self.q_forehead.qsize() > 0:  
                image_cheek.append(self.q_cheek.get())
                image_forehead.append(self.q_forehead.get()) 
                i = i + 1
            
            else:
                time.sleep(0.033)
 

    
########################### cal_bpm ##########################
    
    
    def get_model(self,):
        #모델을 load
        from tensorflow.keras.models import model_from_json
        
        json_file = open('./siamese_model_K.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_json = model_from_json(loaded_model_json)
        
        
        # load weights into new model
        loaded_model_json.load_weights("./siamese_model_weights_K.h5")
                    
        siamese_model = loaded_model_json
        siamese_model.summary()
        
        return siamese_model
        
    #siamese_model -> model
    def siamese_network(self, image_cheek, image_forehead, siamese_model):
        #list를 np array로 바꾸는거 잊지 말기!
        
        import tensorflow as tf
        from tensorflow.keras.models import model_from_json
          
        
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        #from keras import backend as K
        from tensorflow.python.keras import backend as K
        
    
        
        X_test_cheek = np.array(image_cheek)
        X_test_forehead = np.array(image_forehead)
        
        
        for subject in range(0,1):
        
            #temp_cheek = X_test_cheek[subject:subject+1]
            #temp_forehead = X_test_forehead[subject:subject+1]
        
           # ppg_predict = siamese_model.predict([temp_cheek, temp_forehead])
            X_test_cheek = X_test_cheek.reshape(1,600,40,140,3)
            X_test_forehead = X_test_forehead.reshape(1,600,40,140,3)
            predict = siamese_model.predict([X_test_cheek,  X_test_forehead])
        
            ppg = predict[0][0]
            resp = predict[1][0]
                
        return ppg, resp
        
    
    def calculate_bpm(self, ppg):
        #findpeak
        SR = 30
        
        ################## calculate stress ##################
        x, y = signal.periodogram(ppg, SR)

        psd = dict(zip(x, y))

        fx = lambda x: psd[x]

        low_band = []
        high_band = []
        for i in range(len(x)):
            val = x[i]
            if  val >= 0.04 and val < 0.15:
                low_band.append(psd[val])
            elif  val >= 0.15 and val < 0.4:
                high_band.append(psd[val])
            elif val > 0.4:
                break

        lf = math.log(integrate.simps(low_band) + 1)
        hf = math.log(integrate.simps(high_band) + 1)

        self.current_stress = 80*(lf*2/3 + hf/3)
        #print("stress 지수:", self.current_stress)
        
        '''   
        ppg = hp.filtering.smooth_signal(ppg, sample_rate = SR, window_length= 15, polyorder=3)
        ppg = hp.filter_signal(ppg, cutoff = [0.66,3.3], sample_rate = SR, order = 4, filtertype='bandpass')
        plt.plot(ppg)
        '''
        
        w = 5
        ppg = np.convolve(ppg, np.ones(w), 'valid') / w
        ppg = hp.filter_signal(ppg, cutoff = [0.66,3.3], sample_rate = SR, order = 4, filtertype='bandpass')
        
        plt.figure(figsize = (12,4))
        plt.plot(ppg)
        plt.show()
        
        ppg = hp.filtering.smooth_signal(ppg, sample_rate = SR, window_length= 15, polyorder=3)
        #일단 9일 때 제일 잘 됐었음.
        w = 9
        ppg = np.convolve(ppg, np.ones(w), 'valid') / w
        ppg = hp.filtering.smooth_signal(ppg, sample_rate = SR, window_length= 15, polyorder=4)
        
        #plt.show()
        
        #calculate bpm by findpeak
        peaks, _ = find_peaks(ppg, distance=10)
        np.diff(peaks)
        plt.figure(figsize = (12,4))
        plt.plot(ppg)
        plt.plot(peaks, ppg[peaks], "x")
        plt.show()
        
        

        #get average interval between peaks
        interval = 0
        loc = list(peaks)
                
        for i in range(len(loc)-1):
            interval = interval + loc[i+1] - loc[i]
        #interval = interval/(len(max_points)-1)/self.sample_rateFclear
        interval = interval/(len(peaks)-1)/SR
    
        self.current_bpm = 1/interval*60
        
        #print("findpeak bpm:", self.current_bpm)
        
        
      
        
        return self.current_bpm, self.current_stress, ppg
        

    def calculate_rr(self, resp):
        
        SR = 30
        resp = hp.filter_signal(resp, cutoff = [0.1,0.4], sample_rate = SR, order = 2, filtertype='bandpass')
        plt.plot(resp)
        plt.show()

        
        #calculate bpm by findpeak
        peaks, _ = find_peaks(resp, distance=10)
        np.diff(peaks)
        
        
        
        
        
        
        #get average interval between peaks
        interval = 0
        loc = list(peaks)
                
        for i in range(len(loc)-1):
            interval = interval + loc[i+1] - loc[i]
        #interval = interval/(len(max_points)-1)/self.sample_rate
        interval = interval/(len(peaks)-1)/SR
    
        self.current_rr = 1/interval*60
        
       # print("findpeak rr:", self.current_rr)
        
        return self.current_rr, resp
        

    def get_bpm(self, image_cheek, image_forehead, bpm_list, rr_list, model, q_bpm, q_rr ,q_stress, q_ppg, q_resp):
        
        
        if len(image_cheek) != 600:
            return
        
        ppg, resp = self.siamese_network(image_cheek, image_forehead, model)

        #f.write("ppg predict end: ")
        
        self.current_bpm, self.current_stress, ppg = self.calculate_bpm(ppg)
        self.current_rr, resp = self.calculate_rr(resp)
        
        q_ppg.put(ppg)
        q_resp.put(resp)
        
        self.current_bpm = round(self.current_bpm) # bpm 반올림
        self.cuurent_rr = round(self.current_rr)
        self.current_stress = round(self.current_stress)
        
        bpm_list.append(self.current_bpm)
        rr_list.append(self.current_rr)
        q_bpm.put(self.current_bpm)
        q_rr.put(round(self.current_rr))
        q_stress.put(self.current_stress)


        

##############################################################    

    
        
    
    def interface(self, q_frame_gui, q_bpm, q_rr ):
       
       
        bpm = str(0)
        rr = str(0)
        
        
        
        
        dets = []
        while(1):
              
            if self.q_interface.qsize() > 0:
                frame = self.q_interface.get()
            else:
                continue
                
            if q_bpm.qsize() > 0:
                bpm = str(q_bpm.get())
                
            if q_rr.qsize() >0:
                rr = str(q_rr.get())
            
            if self.detect_face.value == 0:
                string = "Can't detect Face!"
                cv2.putText(frame, string, (200,240), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255),thickness = 2) #화면 중앙에 메시지 출력
                bpm = str(0)
                rr = str(0)
                while not q_frame_gui.empty():
                    q_frame_gui.get()
                
            if self.q_face_detector.qsize() > 0:
                dets = self.q_face_detector.get()

           
            for face in dets:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)

            
            bpm_text = "bpm : " + bpm
            rr_text = "rr : " + rr
            #cv2.putText(frame, bpm_text, (20,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            #cv2.putText(frame, rr_text, (20,150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            #cv2.imshow('interface',frame)
            #self.gui.UpdateImage(frame)
            q_frame_gui.put(frame)
            
            key = cv2.waitKey(1)
            if(key == 27):
                break
            
    


##############################################################

    def start(self, q_frame_gui, q_bpm, q_rr, q_stress, q_ppg, q_resp):
        
    
        #여기서 이미지를 가져오고 ppg 예측 시작함. (main)
        
        current_process = psutil.Process()
       
        
        #logger = log_to_stderr()
        #logger.setLevel(multiprocessing.SUBDEBUG)
        def on_press(key): 
            #print('Key %s pressed' % key) 
            pass
            
        def on_release(key): 
            #print('Key %s released' %key) 
            if key == keyboard.Key.esc: #esc 키가 입력되면 종료
                children = current_process.children(recursive=True)
                for child in children:
                    print('Child pid is {}'.format(child.pid)) 
                    child.send_signal(signal.SIGTERM)
                
                current_process.send_signal(signal.SIGTERM)
                    
                while not self.q_frame.empty():
                    self.q_frame.get()
                self.q_frame.close()
                    
                while not self.q_cheek.empty():
                    self.q_cheek.get()
                self.q_cheek.close()
                
                while not self.q_forehead.empty():
                    self.q_forehead.get()
                self.q_forehead.close()
                
                while not self.q_interface.empty():
                    self.q_interface.get()
                self.q_interface.close()
                
                while not q_bpm.empty():
                    q_bpm.get()
                q_bpm.close()
                
                while not self.q_face_detector.empty():
                    self.q_face_detector.get()
                self.q_face_detector.close()
                
                
                
                
                return False # 리스너 등록방법1 
            
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            
        
            p1 = Process(target = self.webcam_get_image, args = ())     
            p2 = Process(target = self.ROI_image, args = ())
            t1 = threading.Thread(target = self.interface, args = (q_frame_gui, q_bpm, q_rr, ))
            
            p1.start()
            p2.start()
            t1.start()
            
            
            count = 0
            
            with Manager() as manager:
                image_cheek = manager.list()
                image_forehead = manager.list()
                
                bpm_list = manager.list()
                rr_list = manager.list()
                
               # f = open("bpm_log.txt", "w")
    
                #get 600 frame image
                for i in range(4):
                    p3 = Process(target = self.get_150_frame, args = ( image_cheek, image_forehead))
                    p3.start()             
                    p3.join()
    
                model = self.get_model()        
        
                while(1):   #조건문 나중에 조작해서 입력 키 값으로 esc같은 거 들어오면 끝내게 만들기
                    
                    
                   #이거 시작하기 전에 deamon = false 같은거 써서 child process만들 수 있게 하기
                    
                    image_cheek_np = np.array(image_cheek)
                    image_forehead_np = np.array(image_forehead)
                    
                    
                    if len(image_cheek) == 600:
                        image_cheek = manager.list(image_cheek[150:600])
                        image_forehead = manager.list(image_forehead[150:600])
                
                        
                    #print("image_cheek_np:", image_cheek_np)
                    #p3 = Process(target = self.get_bpm, args = (image_cheek_np, image_forehead_np, bpm, model))
                    t3 = threading.Thread(target = self.get_bpm, args = (image_cheek_np, image_forehead_np, bpm_list, rr_list, model, q_bpm, q_rr ,q_stress, q_ppg, q_resp))
                    
    
                    p4 = Process(target = self.get_150_frame, args = (image_cheek, image_forehead))
                 
                    #p3.start()
                    t3.start()
                    p4.start()
                    
                    if kb.is_pressed('Esc'):
                        print("end1!!")
                        break
                    
                    #p3.join()
                    t3.join()
                    p4.join()
                    
                               
                    #print(self.current_bpm)
                    print(bpm_list)
                    print(rr_list)
                    #print(len(image_cheek))
                    print("q_frame: ", self.q_frame.qsize())
                    print("q_cheek: ",self.q_cheek.qsize())
                    print("q_interface: ",self.q_interface.qsize())
                    print("q_bpm: ",q_bpm.qsize())
                    print("q_rr: ", q_rr.qsize())
                    print("q_face_detector: ", self.q_face_detector.qsize())
                    print("face detect: ", self.detect_face.value)
                    
                    print("\n")
                    '''
                    temp = bpm[count]
                    string = str(temp)
                    f.write(string)
                    f.write("\n")
                    count = count + 1
                    '''
                    
                    if len(bpm_list) != 0:
                    
                        with open("./data_log.txt", "w") as f: 
                            # write elements of list 
                            #for number in bpm: 
                                #f.write('%s\n' %number)
                            time_str=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                            f.write(time_str)
                            #f.write(str(time.time()))
                            f.write("/")
                            f.write(str(bpm_list[count]))
                            f.write("/")
                            f.write(str(round(rr_list[count])))
                            #f.write("\n")
                        count = count + 1
                    
                    
                    if kb.is_pressed('Esc'):
                        print("end2!!")
                        sys.exit(0)
                        #break
                
                    
                
                print("while문 탈출")
                p1.join()
                #p1.terminate()
                print("p1 끝남")
                p2.join()
                #p2.terminate()
                print("p2 끝남")
                #t1.terminate()
                t1.join()
                
                listener.join()    
                
                bpm_list = np.array(bpm_list)
                rr_list = np.array(rr_list)
                
                return bpm_list, rr_list


def producer(q):
    proc = mp.current_process()
    print(proc.name)

    while True:
        now = datetime.datetime.now()
        data = str(now)
        q.put(data)
        time.sleep(1)


class Consumer(QThread):
    poped = pyqtSignal(str)
    poped2 = pyqtSignal(QImage)


    def __init__(self, q_frame_gui, q_bpm, q_rr, q_stress):
        super().__init__()
        self.q_frame_gui = q_frame_gui
        self.q_bpm = q_bpm
        self.q_rr = q_rr
        self.q_stress = q_stress


    def run(self):     
        while True:
            if not self.q_frame_gui.empty():
                data_frame = self.q_frame_gui.get()
                
                image = cv2.cvtColor(data_frame, cv2.COLOR_BGR2RGB)
                h,w,c = image.shape
                qimage = QtGui.QImage(image.data, w,h,w*c, QtGui.QImage.Format_RGB888)
                #print("qimage signal emit")
                self.poped2.emit(qimage)
                
            if not(self.q_bpm.empty() and self.q_rr.empty() and self.q_stress.empty()):
                data_bpm = q_bpm.get()
                data_rr = q_rr.get()
                data_stress = q_stress.get()
                #print("str signal emit")
                self.poped.emit(str(data_bpm) +'/'+ str(data_rr) + '/' + str(data_stress))
                
class Consumer_Graph(QThread):
    poped = pyqtSignal(list)
    poped2 = pyqtSignal(list)


    def __init__(self, q_ppg, q_resp):
        super().__init__()
        
        self.q_ppg = q_ppg
        self.q_resp = q_resp

    def run(self):     
        while True:
            if (self.q_ppg.empty()==False and self.q_resp.empty()==False):
                data_ppg = q_ppg.get()
                data_resp = q_resp.get()
                
                #print("ppg signal emit")
                self.poped.emit(data_ppg.tolist())
                #print("resp signal emit")
                self.poped2.emit(data_resp.tolist())  


class loading(QWidget, form_widget):
    def __init__(self, parent):
        super(loading,self).__init__(parent)
        self.setupUi(self)
        self.center()
       
        self.show()
        
        self.movie = QMovie('./loading.gif', QByteArray(), self)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.label.setMovie(self.movie)
        
        self.set_transparent(0.5)
  
        self.movie.start()
        
        
    def center(self):
        size = self.size()
        ph = self.parent().geometry().height()
        pw =  self.parent().geometry().width()
        self.move(int(pw/2 - size.width()/2), int(ph/2 - size.height()/2))
        
    def set_transparent(self,opacity):
        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(opacity)
        self.setGraphicsEffect(opacity_effect)


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(211, xlim=(0, 150), ylim=(-2, 2))
        self.axes2 = fig.add_subplot(212, xlim=(0, 150), ylim=(-1, 1))

        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
    def compute_initial_figure(self):
        pass
    
    

class AnimationWidget(QWidget):
    def __init__(self, q_ppg, q_resp):
        QMainWindow.__init__(self)
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self, width=5, height=4, dpi=100)
        vbox.addWidget(self.canvas)
        hbox = QHBoxLayout()
        

        
        self.ppg = np.zeros(100, dtype=np.float)
        self.resp = np.zeros(100, dtype=np.float)
        self.consumer = Consumer_Graph(q_ppg, q_resp)
        self.consumer.poped.connect(self.get_data_bpm)
        self.consumer.poped2.connect(self.get_data_resp)
        self.consumer.start()
        
        
        self.x = np.arange(150)
        self.y = np.ones(150, dtype=np.float)*np.nan
        self.line, = self.canvas.axes.plot(self.x, self.y, animated=True, color='red', lw=2)

        self.x2 = np.arange(150)
        self.y2 = np.ones(150, dtype=np.float)*np.nan
        self.line2, = self.canvas.axes2.plot(self.x2, self.y2, animated=True, color='blue', lw=2)
        
        self.on_start()


    def update_line(self, i):
    
        if len(self.ppg) < 150 and i!=0:
            return [self.line]
        
        old_y = self.line.get_ydata()
        new_index = self.ppg[i]
        new_y = np.r_[old_y[1:], new_index]
        self.line.set_ydata(new_y)
        
        if i == 149:
            self.ppg = self.ppg[150:]
        
        return [self.line]

        
    def update_line2(self, i):
        if len(self.resp) < 150 and i != 0:
            return [self.line]

        old_y2 = self.line2.get_ydata()
        new_index2 = self.resp[i]
        new_y2 = np.r_[old_y2[1:], new_index2]
        self.line2.set_ydata(new_y2)
        
        if i == 149:
            self.resp = self.resp[150:]
        
        return [self.line2]

    
    @pyqtSlot(list)  
    def get_data_bpm(self, data_ppg):
        
        ppg = np.array(data_ppg)
        self.ppg = np.append(self.ppg, ppg)
        
        #print("graph data emit bpm")
        
    @pyqtSlot(list)
    def get_data_resp(self, data_resp):
        resp = np.array(data_resp)
        self.resp = np.append(self.resp, resp)
        #print("graph data emit resp")
   

        
    def on_start(self):
        self.ani = animation.FuncAnimation(self.canvas.figure, self.update_line,frames=150, blit=True, interval=10)
        self.ani2 = animation.FuncAnimation(self.canvas.figure, self.update_line2,frames=150, blit=True, interval=10)
                   


class MyWindow(QMainWindow, form_class):
    def __init__(self, q_frame_gui, q_bpm, q_rr, q_stress):
        super().__init__()
        self.setupUi(self)
        #self.pushButton.clicked.connect(self.btn_clicked)

        # thread for data consumer
        self.consumer = Consumer(q_frame_gui, q_bpm, q_rr, q_stress)
        
        self.consumer.poped.connect(self.print_data)
        self.consumer.poped2.connect(self.UpdateVideo)
        self.consumer.start()
        self.setWindowTitle("너의 맥박이 보여")
        self.setWindowIcon(QIcon('./icon.jpg'))
        
        
        self.loading = loading(self)
        self.is_calculating = 1
        
        
    #def btn_clicked(self):
      # QMessageBox.about(self, "START", "clicked")

    @pyqtSlot(str)
    def print_data(self, data):
        #print("str signal pyqtslot")
        data_split = data.split("/")
       
        self.label_2.setText("BPM: " + data_split[0]) # BPM
        self.label_3.setText("RPM: " + data_split[1]) # RPM
        self.label_4.setText("Stress: " + data_split[2]) # Stress
        
        if self.is_calculating == 1:
            self.is_calculating = 0
            self.loading.hide()
        
        #self.label_5.setPixmap(QtGui.QPixmap("C:/Deep/Realtime_TEST/중요/heart.png")) #image path


        
    @pyqtSlot(QImage)
    def UpdateVideo(self, qimage):
        #print("QImage signal pyqtslot")
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.label.setPixmap(pixmap)
        #self.label.update()
        

        '''
        
        if self.is_calculating == 0 and :
            self.is_calculating = 1
            self.loading.show()
            '''

    
if __name__ == "__main__":
    q_bpm = Queue()
    q_rr = Queue()
    q_stress = Queue()
    q_frame_gui = Queue(maxsize = 10000) 
    
    q_ppg = Queue()
    q_resp = Queue()
    
    ppg_predict = RealTime_PPG_Predict()

    # producer process
    p = Process(name="producer", target= ppg_predict.start, args=(q_frame_gui, q_bpm,q_rr, q_stress, q_ppg, q_resp), daemon=False)
    #print("process created")

    p.start()
    #print("process start!")


    # Main process
    app = QApplication(sys.argv)
    mywindow = MyWindow(q_frame_gui, q_bpm, q_rr, q_stress)
    mywindow.show()
    aw = AnimationWidget(q_ppg, q_resp)
    aw.show()
    sys.exit(app.exec_())
    
    p.join()
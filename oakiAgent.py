#!/usr/bin/env python3

## import packages Computer vision
import cv2
import mediapipe as mp
import depthai as dai
import blobconverter

## Methods packages
from methods.utils import *
from methods.body_part_angle import BodyPartAngle
from methods.types_of_exercise import TypeOfExercise

## General packages
import imutils
import time
#import simpleaudio as sa
from playsound import playsound
import threading
import numpy as np
import multiprocessing
from multiprocessing import Process

class oaki():
    def __init__(self):
        '''
        Init variables
        '''

        #threading.Thread(target=self.sound, args=(1,)).start()
        # Create self.pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutRGB = self.pipeline.create(dai.node.XLinkOut) 

        xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
        xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

        xoutDepth.setStreamName("depth")
        xoutRGB.setStreamName("rectifiedLeft") 

        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        lrcheck = False
        subpixel = False

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(lrcheck)
        stereo.setSubpixel(subpixel)

        # Config Init
        topLeft = dai.Point2f(0.4, 0.4)
        bottomRight = dai.Point2f(0.6, 0.6)

        self.config = dai.SpatialLocationCalculatorConfigData()
        self.config.depthThresholds.lowerThreshold = 100
        self.config.depthThresholds.upperThreshold = 10000
        self.config.roi = dai.Rect(topLeft, bottomRight)

        spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        spatialLocationCalculator.initialConfig.addROI(self.config)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
        stereo.depth.link(spatialLocationCalculator.inputDepth)


        stereo.rectifiedLeft.link(xoutRGB.input) 


        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

        # MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.exercise_type = 'push-up'

        # load Oaki
        self.oaki_img = cv2.imread("img/OAKI.png", cv2.IMREAD_COLOR)
        self.bBandera = True
        self.process_count = 0
        self.phrase = {
            0: "Hi Welcome, my name is OAKI, and I'll be your trainer for today", 
            1: "Let's start with some stretch",
            2: "Go back! You have to be 3 meters from the camera",
            3: "You know I can measure the depth from the camera to your body",
            4: "Too much, please come closer",
            5: "Now it's perfect, please put your hands up",
            6: "Now put your hands forward",
            7: "Now let's start the training",
            8: "Today we are going to do some push-ups",
            9: "Five more",
            10: "The last one",
            11: "Good Job, we are done for today",
            12: "See you tomorrow Adonai",
        }
        self.q1 = multiprocessing.Queue()
        p = Process(target=self.sound, args=(self.q1,))
        p.start()

    def addImageWatermark(waterImg,OriImg,pos=(10,100),):
        tempImg = OriImg.copy()
        overlay = transparentOverlay(tempImg, waterImg, pos)
        output = OriImg.copy()
        # apply the overlay
        cv2.addWeighted(overlay, output, 1 - opacity, 0, output)
        cv2.imshow('Life2Coding', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sound(self, queue):
        time.sleep(5)
        while True:
            nCount = queue.get()

            filename = "voice/{}.mp3".format(nCount)
            playsound(filename)
            time.sleep(2)


    def run(self):
        ## setup mediapipe
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:           
            # movement of exercise
            counter = 0  

            # state of move
            status = True  
            start_training = 0
            bProcess0 = 0
            bProcess1 = 0
            bProcess2 = 0
            bProcess3 = 0
            bProcess4 = 0
            bProcess5 = 0
            bProcess6 = 0
            bProcess7 = 0
            bProcess8 = 0
            bProcessX = 0
            bProcessY = 0
            bProcessW = 0
            # Connect to device and start self.pipeline
            with dai.Device(self.pipeline) as device:

                # Output queue will be used to get the depth frames from the outputs defined above
                depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                rgbQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=4, blocking=False)
                spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
                spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

                color = (255, 255, 255)

                while True:

                    
                    if self.bBandera:
                        self.oaki_img_reduce = cv2.resize(self.oaki_img,(1280, 720))
                        
                        self.bBandera= False
                    self.oaki_img_save = self.oaki_img_reduce.copy()
                    inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

                    depthFrame = inDepth.getFrame()

                    inFrame = rgbQueue.get()

                    frame = inFrame.getCvFrame()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ## make detection
                    results = pose.process(frame)
                    (imgH, imgW) = frame.shape[:2]

                    # Get the depth to the new object 
                    '''
                    config.roi = dai.Rect(topLeft, bottomRight)
                    cfg = dai.SpatialLocationCalculatorConfig()
                    cfg.addROI(config)
                    spatialCalcConfigInQueue.send(cfg)
                    '''
                    try:
                        landmarks = results.pose_landmarks.landmark
                        counter, status = TypeOfExercise(landmarks).calculate_exercise(
                            self.exercise_type, counter, status)

                        normalizedLandmarkRight = detection_body_part(landmarks, "RIGHT_HIP")
                        normalizedLandmarkLeft = detection_body_part(landmarks, "LEFT_HIP")
                        normalizedLandmarkShoulder = detection_body_part(landmarks, "LEFT_SHOULDER")

                        pixelCoordinatesLandmarkRight = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmarkRight[0], normalizedLandmarkRight[1], imgW, imgH)
                        pixelCoordinatesLandmarkLeft = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmarkShoulder[0], normalizedLandmarkShoulder[1], imgW, imgH)
                        pixelCoordinatesLandmarkShoulder = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmarkLeft[0], normalizedLandmarkLeft[1], imgW, imgH)


                        topLx = normalizedLandmarkRight[0]
                        topLy = normalizedLandmarkShoulder[1]

                        bottomRx = normalizedLandmarkLeft[0]
                        bottomRy = normalizedLandmarkLeft[1]
                        

                        # Optimize later
                        if topLx < 0:
                            topLx = 0
                        if topLy < 0:
                            topLy = 0 
                        if bottomRx < 0:
                            bottomRx = 0
                        if bottomRy < 0:
                            bottomRy = 0  

                        topLeft = dai.Point2f(topLx, topLy)
                        bottomRight = dai.Point2f(bottomRx, bottomRy)

                        self.config.roi = dai.Rect(topLeft, bottomRight)
                        cfg = dai.SpatialLocationCalculatorConfig()
                        cfg.addROI(self.config)
                        spatialCalcConfigInQueue.send(cfg)

                    except Exception as error:
                        pass

                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255),
                                            thickness=2,
                                            circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(174, 139, 45),
                                            thickness=2,
                                            circle_radius=2),
                    )

                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)


                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                    spatialData = spatialCalcQueue.get().getSpatialLocations()
                    for depthData in spatialData:
                        roi = depthData.config.roi
                        roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                        depthMin = depthData.depthMin
                        depthMax = depthData.depthMax


                    # Dialogue Structure (I can improve this) | Sequential training. Simple v1
                    if self.process_count == 0:
                        self.process_count = 0
                        if start_training == 0:
                            self.q1.put(self.process_count)
                        start_training += 1
                    if self.process_count < 2 and start_training > 100:
                        self.process_count = 1
                        if bProcess0 == 0:
                            self.q1.put(self.process_count)
                        bProcess0 += 1
                    if  float(depthData.spatialCoordinates.z/10) < 220 and self.process_count < 3 and bProcess0 > 100:

                        self.process_count = 2

                        if bProcessX == 0:
                            self.q1.put(self.process_count)
                        bProcessX += 1
                    if  float(depthData.spatialCoordinates.z/10) < 220 and self.process_count < 3 and bProcessX > 100:
                        self.process_count = 3

                        if bProcessY == 0:
                            self.q1.put(self.process_count)
                        bProcessY += 1

                    if self.process_count < 5 and float(depthData.spatialCoordinates.z/10) > 320:
                        self.process_count = 4
                        if bProcessW == 0:
                            self.q1.put(self.process_count)
                        bProcessW += 1
                    if self.process_count < 6 and float(depthData.spatialCoordinates.z/10) < 320 and float(depthData.spatialCoordinates.z/10) > 220 and bProcessW > 50:
                        self.process_count = 5
                        if bProcess1 == 0:
                            self.q1.put(self.process_count)
                        bProcess1 += 1
                    if self.process_count < 7 and counter < 5 and bProcess1 > 100:
                        self.process_count = 6
                        if bProcess2 == 0:
                            self.q1.put(self.process_count)
                        bProcess2 += 1
                    if self.process_count < 8 and counter < 5 and bProcess2 > 100:
                        self.process_count = 7
                        if bProcess3 == 1:
                            self.q1.put(self.process_count)
                        bProcess3 += 1
                        counter = 0
                    if self.process_count < 9 and counter <5 and bProcess3 > 100:
                        self.process_count = 8
                        if bProcess4 == 0:
                            self.q1.put(self.process_count)
                        bProcess4 = 1
                    if self.process_count < 10 and counter > 10:
                        self.process_count = 9
                        if bProcess5 == 0:
                            self.q1.put(self.process_count)
                        bProcess5 = 1
                    if self.process_count < 11 and counter > 13:
                        self.process_count = 10
                        if bProcess6== 0:
                            self.q1.put(self.process_count)
                        bProcess6 += 1
                    if self.process_count < 12 and counter > 14:
                        self.process_count = 11
                        if bProcess7 == 0:
                            self.q1.put(self.process_count)
                        bProcess7 += 1
                    if self.process_count < 13 and bProcess7 > 200:
                        self.process_count = 12
                        if bProcess8 == 0:
                            self.q1.put(self.process_count)
                        bProcess8 = 1






                    """
                    self.phrase = {
        
                    2: "Go back! You have to be 3 meters from the camera",
                    3: "You know I can measure the depth from the camera to your body",
                    4: "Too much, please come closer",
                    5: "Now it's perfect, please put your hands up",
                    6: "Now put your hands forward",
                    7: "Now let's start the training",
                    8: "Today we are going to do some push-ups",
                    9: "Five more",
                    10: "The last one",
                    11: "Good Job, we are done for today",
                    12: "See you tomorrow Adonai",
                }
                    """



                    y_offset = 50
                    x_offset = 100
                    self.oaki_img_save[y_offset:y_offset+frame.shape[0], x_offset:x_offset+frame.shape[1]] = frame
                    
                    xmin = 900
                    ymin = 20


                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    
                    cv2.putText(self.oaki_img_save, f"Analytics Training", (xmin + 10, ymin + 30), fontType, 1, (255, 255, 255))
                    cv2.putText(self.oaki_img_save, f"X: {int(depthData.spatialCoordinates.x/10)} cm", (xmin + 10, ymin + 60), fontType, 1, (255, 255, 255))
                    cv2.putText(self.oaki_img_save, f"Y: {int(depthData.spatialCoordinates.y/10)} cm", (xmin + 10, ymin + 90), fontType, 1, (255, 255, 255))
                    cv2.putText(self.oaki_img_save, f"Z: {int(depthData.spatialCoordinates.z/10)} cm", (xmin + 10, ymin + 120), fontType, 1, (255, 255, 255))
                    cv2.putText(self.oaki_img_save, f"Repeticiones: {counter}", (xmin + 10, ymin + 150), fontType, 1, (255, 255, 255))
                    
                    
                    text_print = self.phrase[self.process_count]
                    img = cv2.putText(self.oaki_img_save, text_print, (int(self.oaki_img_save.shape[1]/25), int(self.oaki_img_save.shape[0]/1.25)), fontType, 0.7, (255, 255, 255))
                    
               
                    # Show the frame
                    cv2.imshow("depth", self.oaki_img_save)
                    
                    
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

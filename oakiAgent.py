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
from playsound import playsound
import threading
import numpy as np


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
        self.oaki_img = cv2.imread("img/oaki-happy.jpg", cv2.IMREAD_COLOR)
        self.bBandera = True
        

    def addImageWatermark(waterImg,OriImg,pos=(10,100),):
        tempImg = OriImg.copy()
        overlay = transparentOverlay(tempImg, waterImg, pos)
        output = OriImg.copy()
        # apply the overlay
        cv2.addWeighted(overlay, output, 1 - opacity, 0, output)
        cv2.imshow('Life2Coding', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sound(self, nType):
        path = "voice/{}.mp3".format(nType)
        playsound(path)

    def run(self):
        ## setup mediapipe
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:           
            # movement of exercise
            counter = 0  

            # state of move
            status = True  

            # Connect to device and start self.pipeline
            with dai.Device(self.pipeline) as device:

                # Output queue will be used to get the depth frames from the outputs defined above
                depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                rgbQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=4, blocking=False)
                spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
                spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

                color = (255, 255, 255)

                while True:
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

                        normalizedLandmark = detection_body_part(landmarks, "NOSE")
                        
                        pixelCoordinatesLandmarkTip = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark[0], normalizedLandmark[1], imgW, imgH)

                        topLx = pixelCoordinatesLandmarkTip[0]-10
                        topLy = pixelCoordinatesLandmarkTip[1]-10

                        bottomRx = pixelCoordinatesLandmarkTip[0]+10
                        bottomRy = pixelCoordinatesLandmarkTip[1]+10
                        
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

                    
                    img = np.zeros([int(frame.shape[0]/4),frame.shape[1]-100,3],dtype=np.uint8)
                    img.fill(255) 

                    xmin = 5
                    ymin = 5
                    xmax = 200
                    ymax = 95

                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1)
                    cv2.putText(img, f"Analytics Training", (xmin + 10, ymin + 15), fontType, 0.5, (0, 0, 0))
                    cv2.putText(img, f"X: {int(depthData.spatialCoordinates.x/100)} cm", (xmin + 10, ymin + 30), fontType, 0.5, (0, 0, 0))
                    cv2.putText(img, f"Y: {int(depthData.spatialCoordinates.y/100)} cm", (xmin + 10, ymin + 45), fontType, 0.5, (0, 0, 0))
                    cv2.putText(img, f"Z: {int(depthData.spatialCoordinates.z/100)} cm", (xmin + 10, ymin + 60), fontType, 0.5, (0, 0, 0))
                    cv2.putText(img, f"Repeticiones: {counter}", (xmin + 10, ymin + 75), fontType, 0.5, (0, 0, 0))
                    
                    self.text = "tu puedes "
                    text_print = "Good Job"
                    img = cv2.putText(img, text_print, (int(frame.shape[1]/2), int(frame.shape[0]/8)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    if self.bBandera:
                        self.oaki_img_reduce = cv2.resize(self.oaki_img,(100, int(frame.shape[0]/4)))
                        
                        self.bBandera= False
                    
                    # we add the logo
                    frame_logo = np.hstack((img,self.oaki_img_reduce))

                    # Save the results in the image
                    frame_results = np.vstack((frame,frame_logo))
                    
                    # Show the frame
                    cv2.imshow("depth", frame_results)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

## import packages Computer vision
import cv2
import mediapipe as mp
import depthai
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


def sound(nType):
    path = "voice/{}.mp3".format(nType)
    playsound(path)


def start_training():

    #threading.Thread(target=sound, args=(1,)).start()

    # Global variables
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = depthai.Pipeline()

    # First, we want the Color camera as the output
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
    cam_rgb.setInterleaved(False)

    # XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
    xout_rgb = pipeline.createXLinkOut()
    # For the rgb camera output, we want the XLink stream to be named "rgb"
    xout_rgb.setStreamName("rgb")
    # Linking camera preview to XLink input, so that the frames will be sent to host
    cam_rgb.preview.link(xout_rgb.input)

    exercise_type = 'squat'

    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:

        counter = 0  # movement of exercise
        status = True  # state of move
        depth_status= True
        depth_status_hands = False
        with depthai.Device(pipeline) as device:
            # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
            q_rgb = device.getOutputQueue("rgb")

            # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
            frame = None

            # Main host-side application loop
            #threading.Thread(target=sound, args=(2,)).start()
            while True:

                # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
                in_rgb = q_rgb.tryGet()

                if in_rgb is not None:
                    # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                    frame = in_rgb.getCvFrame()

                if frame is not None:
                    frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
                    ## recolor frame to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False
                    ## make detection
                    results = pose.process(frame)
                    ## recolor back to BGR
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    try:
                        landmarks = results.pose_landmarks.landmark
                        counter, status = TypeOfExercise(landmarks).calculate_exercise(
                            exercise_type, counter, status)
                    except:
                        pass

                    depth_oak = 3
                    '''
                    if depth_oak > 1.5 and depth_oak < 2 and depth_status == True:
                        threading.Thread(target=sound, args=(5,)).start()
                        threading.Thread(target=sound, args=(6,)).start()
                    elif depth_oak < 1.5:
                        threading.Thread(target=sound, args=(3,)).start()
                        depth_status_hands = True
                    elif depth_oak < 2:
                        threading.Thread(target=sound, args=(4,)).start()
                    '''

                    depth_hands = 3
                    '''
                    if depth_hands > 1.5 and depth_status_hands == True:

                        threading.Thread(target=sound, args=(7,)).start()
                    else:
                        threading.Thread(target=sound, args=(8,)).start()
                        threading.Thread(target=sound, args=(9,)).start()
                    '''
                    
                    if counter == 10:
                        threading.Thread(target=sound, args=(10,)).start()

                    if counter == 14:
                        threading.Thread(target=sound, args=(11,)).start()
                        break


                    #score_table(exercise_type, counter, status)

                    ## render detections (for landmarks)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255),
                                            thickness=2,
                                            circle_radius=2),
                        mp_drawing.DrawingSpec(color=(174, 139, 45),
                                            thickness=2,
                                            circle_radius=2),
                    )
                    cv2.imshow("prueba", frame)
                
            
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            #threading.Thread(target=sound, args=(12,)).start()
            #threading.Thread(target=sound, args=(13,)).start()
            time.sleep(10)
            cap.release()
            cv2.destroyAllWindows()
        

if __name__ == "__main__" :
    start_training()
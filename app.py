## import packages
import cv2
from methods.utils import *
import mediapipe as mp
from methods.body_part_angle import BodyPartAngle
from methods.types_of_exercise import TypeOfExercise
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

def init_video():
    global cap
    global mp_drawing
    global mp_pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)#)
    cap.set(3, 800)  # width
    cap.set(4, 480)  # height
    show_video()

def show_video():
    # Global variables
    global cap
    global mp_drawing
    global mp_pose

    exercise_type = 'push-up'

    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:

        counter = 0  # movement of exercise
        status = True  # state of move
        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)
            if ret == True:
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
                #cv2.imshow("prueba", frame)
                
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                
                lblVideo.after(10, show_video)
            else:
                lblVideo.image = ""
                cap.release()
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
       

def stop_video():
    global cap
    cap.release()




cap = None
root = Tk()
btninit_video = Button(root, text="Start Training", width=45, command=init_video)
btninit_video.grid(column=0, row=0, padx=5, pady=5)
btnstop_video = Button(root, text="Stop Training", width=45, command=stop_video)
btnstop_video.grid(column=1, row=0, padx=5, pady=5)
lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)
root.mainloop()
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

import threading

from scipy.spatial import distance

import numpy as np

import time 

# eye: vertical cordinates
# horziontal: horizontal cordinates 
# left_or_right: 0 - right 1-left

def calculate_EAR(eye, horizontal, left_or_right):
    cumRatio = 0

    C = distance.euclidean(horizontal[0], horizontal[1])

    if left_or_right == 0:
        for x in range(0, 6):
            cumRatio += distance.euclidean(eye[x], eye[x+6])
    
    else:
        cumRatio += distance.euclidean(eye[0], eye[12])
        cumRatio += distance.euclidean(eye[1], eye[9])
        cumRatio += distance.euclidean(eye[2], eye[8])
        cumRatio += distance.euclidean(eye[3], eye[7])
        cumRatio += distance.euclidean(eye[4], eye[6])
        cumRatio += distance.euclidean(eye[5], eye[11])
        cumRatio += distance.euclidean(eye[6], eye[10])
    
    ear_aspect_ratio = cumRatio/(7.0*C)
    return ear_aspect_ratio

def calculate_MAR(mouth):
    A = distance.euclidean(mouth[1], mouth[11])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[3], mouth[9])
    D = distance.euclidean(mouth[4], mouth[8])
    E = distance.euclidean(mouth[0], mouth[6])
    
    mouth_aspect_ratio = (A+B+C+D)/(4.0* E)
    return mouth_aspect_ratio

def runner():
    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            start = time.time()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
            
            leftEye = []
            rightEye = []
            mouth = []
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    leftEye = []
                    rightEye = []
                    mouth = []
                    
                    leftlower =[]
                    leftupper = []
                    
                    rightlower =[]
                    rightupper = []
                    
                    leftHorizontal = []
                    rightHorizontal = []
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 249 or idx == 390 or idx == 373 or idx == 374 or idx == 380 or idx == 381 or idx == 382:
                        leftlower.append((lm.x, lm.y))
                    if id == 466 or idx ==  388 or idx == 387 or idx == 386 or idx == 385 or idx == 384 or idx == 398:
                        leftupper.append((lm.x, lm.y))
                    
                    if idx == 7 or idx == 163 or idx == 144 or idx == 145 or idx == 153 or idx == 154 or idx == 155:
                        rightlower.append((lm.x, lm.y))
                    if id == 246 or idx ==  161 or idx == 160 or idx == 159 or idx == 158 or idx == 157 or idx == 173:
                        rightupper.append((lm.x, lm.y))
                    
                    if idx == 263 or idx == 362:
                        leftHorizontal.append((lm.x, lm.y))
                        
                    if idx == 33 or idx == 133:
                        rightHorizontal.append((lm.x, lm.y))
                    
                
                    
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:  
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z]) 

                
                leftEye = leftlower + leftupper
                leftEyeRatio = calculate_EAR(leftEye, leftHorizontal, 0)
                # print(leftEyeRatio)
                
                
                # verify this again in calculate_EAR
                rightEye = rightlower + rightupper
                rightEyeRatio = calculate_EAR(rightEye, rightHorizontal, 1)
                # print(rightEyeRatio)
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]]).reshape((3,3))
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"
                    
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()

if __name__ == "__main__":
    t1 = threading.Thread(target=runner, args=())

    t1.start()
    t1.join()
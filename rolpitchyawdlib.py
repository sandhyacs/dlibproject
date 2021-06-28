import cv2
import dlib
import numpy as np
import math


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)
    print(landmarks)
    image_points = np.array([
                            (landmarks[2], landmarks[3]),     # Nose tip
                            (landmarks[0], landmarks[1]),   # Chin
                            (landmarks[4], landmarks[5]),     # Left eye left corner
                            (landmarks[6], landmarks[7]),     # Right eye right corne
                            (landmarks[8], landmarks[9]),     # Left Mouth corner
                            (landmarks[10], landmarks[11])      # Right mouth corner
                        ], dtype="double")

    

                        
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[2], landmarks[3])

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    
    return image_with_landmarks, landmarks

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 

while True:
    ret, frame = cap.read()   
    image_landmarks, landmarks = mouth_open(frame)
    land=[]
    flag=0
    name=str(type(landmarks))
    catch=str(name[8:11])
    if catch !="int":
        flag=1
        print(landmarks)
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            print(pos)
            print(idx)
            if idx==30:
                #nose
                land.append(pos[0])
                land.append(pos[1])
            if idx==8:
                #chin
                land.append(pos[0])
                land.append(pos[1])
            if idx==36:
                #lefteye
                land.append(pos[0])
                land.append(pos[1])
            if idx==45:
                #righteye
                land.append(pos[0])
                land.append(pos[1])
            if idx==48:
                #leftmouth
                land.append(pos[0])
                land.append(pos[1])
            if idx==54:
                #rightmouth
                land.append(pos[0])
                land.append(pos[1])
        if flag==1:
            imgpts, modelpts, rotate_degree, nose = face_orientation(frame, land)
            
            cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
            cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED
            for j in range(len(rotate_degree)):
                cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 

import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0) #This initializes the video capture object using OpenCV (default)
mpHands = mp.solutions.hands #This uses the hands solution in the Mediapip library and provides a pre-trained model for hand tracking
hands = mpHands.Hands() #Initializes the Hands class from the Mediapip 'hands' solution with different parameters
mpDraw = mp.solutions.drawing_utils

while True: #This starts an infinite loop that only stops when told explicitly
    success, img = cam.read()
    #success is a boolean value that indicates if it succeeded
    #img is the captured frame that is and image represented by a NumPy array

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #The prebuilt model only accepts RGB images so we convert it to RGB
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks) 
    #To ensure that our hands are being detected

    if(results.multi_hand_landmarks):
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #displaying original image not RGB one



    #results will then store the results of the processed image
    cv2.imshow("Image",img) #This is used to open a window with the name "Image" using the image read above
    cv2.waitKey(1) #This waits for a key to be pressed for a certain amount of time.
    #If the timer runs out, the program continues
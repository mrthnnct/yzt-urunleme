import cv2

# not being used right now but will 'accelerate' face detection
import tensorrt as trt

# load data into cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 0 is the default video capture device
# for other devices get code and enter into videoCapture as a parameter
# why is this ai this good at commenting
cap = cv2.VideoCapture(0)

while True:
    # grabs, decodes and returns next frame of video
    ret, frame = cap.read() # can also be written as .grab() and .retrieve()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    faces = face_cascade.detectMultiScale(
        gray, # since cascade works in greyscale
        scaleFactor=1.1,
        minNeighbors=5, # all parameters here can be edited to improve accuracy and/or speed
        minSize=(30, 30)
    )
    
    # detected faces return a tuple of values for the rectangle found around the image
    # that's why we iterate thru the tuple values in faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # show frame
    cv2.imshow('Face Detection', frame)
    
    # exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clear resources
cap.release()
cv2.destroyAllWindows()
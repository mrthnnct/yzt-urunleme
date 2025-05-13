import cv2
import numpy as np
import os
import sys
import torch
import math
import mediapipe as mp
import face_recognition
import pickle
from datetime import datetime

## mediapipe is now implemented correctly


## TODO last committeki 7 gazilyon yeni satırı pull reqde açıkla düzgünce yoksa mal gibi durur ty

# Set environment variable to prefer NVIDIA GPU
os.environ["OPENCV_DNN_BACKEND_CUDA"] = "1"
os.environ["OPENCV_OPENCL_DEVICE"] = "NVIDIA"
os.environ["OPENCV_VIDEOIO_BACKEND_CUDA"] = "CUDA"

print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_device(0)  # Use first GPU

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def check_cuda():
    try:
        # Check OpenCV CUDA
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"OpenCV CUDA device count: {device_count}")
        
        # Check PyTorch CUDA
        pytorch_cuda = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {pytorch_cuda}")
        
        # If either is available, return True
        if device_count > 0 or pytorch_cuda:
            if device_count > 0:
                cv2.cuda.setDevice(0)
                cv2.setUseOptimized(True)
                try:
                    cv2.cuda.printCudaDeviceInfo(0)
                except Exception as e:
                    print(f"Warning: {e}")
            
            return True
        return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

# Improved rotation function that minimizes black borders
def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions after rotation
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust the rotation matrix to take into account the new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform the rotation and return the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated_image

# Improved cropping function with better threshold handling
def crop_empty_space(image, threshold=5):
    """
    Crops empty/black space from around an image.
    Args:
        image: The input image
        threshold: Pixel values below this are considered empty/black
    Returns:
        Cropped image without empty borders
    """
    if image.size == 0:
        return image
        
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Find the non-empty pixels with more aggressive threshold
    mask = gray > threshold
    
    # Find the bounding box of non-empty pixels
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add a small margin to ensure we don't cut too tight
    margin = 10
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(image.shape[0] - 1, y_max + margin)
    x_max = min(image.shape[1] - 1, x_max + margin)
    
    # Crop the image to the bounding box
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    return cropped

def detect_face_orientation(face_image):
    """
    Detects face orientation using MediaPipe face mesh landmarks.
    Returns the angle needed to correct to make the face upright.
    """
    try:
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return 0  # No face detected
        
        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get eye landmarks (33=right eye, 263=left eye)
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Get height and width of the image
        h, w = face_image.shape[:2]
        
        # Convert normalized coordinates to pixel values
        left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
        right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
        
        # Calculate angle
        dx = right_eye_x - left_eye_x
        dy = right_eye_y - left_eye_y
        
        # Calculate the angle of the line between the eyes (in degrees)
        # When a face is upright, the eyes should be horizontal (angle close to 0)
        current_angle = math.degrees(math.atan2(dy, dx))
        
        # Return the correction angle needed
        # If the face is tilted, this is the angle we need to apply to make it upright
        return -current_angle
        
    except Exception as e:
        print(f"Error in face orientation detection: {e}")
        return 0  # Default to no rotation if all methods fail
            
def process_face_upright(face_image, detected_angle, display_width, display_height):
    """
    Processes a face image to ensure it's upright and fills the frame
    without black borders.
    """
    if face_image.size == 0:
        return np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # CORRECT ROTATION LOGIC: Apply the exact opposite of the detected angle
    # This is the key fix - we're counteracting the exact tilt of the face
    # For example, if face is tilted 45° clockwise, we rotate 45° counterclockwise
    correction_angle = -detected_angle
    
    # Only rotate if angle is significant
    if abs(correction_angle) > 2:  # Using a slightly higher threshold to avoid micro-rotations
        # Apply rotation correction
        rotated_face = rotate_image(face_image, correction_angle)
        
        # Remove any black borders that appeared after rotation
        rotated_face = crop_empty_space(rotated_face, threshold=5)
    else:
        rotated_face = face_image
    
    # Resize to fill the display window completely with no black borders
    if rotated_face.size > 0:
        # Calculate scaling factors for width and height
        scale_width = display_width / rotated_face.shape[1]
        scale_height = display_height / rotated_face.shape[0]
        
        # Use the LARGER scale to ensure the image fills the window completely
        # Increased to 1.2 (20% extra) to ensure no borders due to rounding
        scale = max(scale_width, scale_height) * 1.2
        
        # Calculate new dimensions
        new_width = int(rotated_face.shape[1] * scale)
        new_height = int(rotated_face.shape[0] * scale)
        
        # Resize the image
        resized_face = cv2.resize(rotated_face, (new_width, new_height))
        
        # Calculate center crop position to focus on middle of face
        start_x = max(0, (new_width - display_width) // 2)
        start_y = max(0, (new_height - display_height) // 2)
        
        # Ensure we don't exceed image bounds
        end_x = min(new_width, start_x + display_width)
        end_y = min(new_height, start_y + display_height)
        
        # Make sure we have enough image to fill the display
        if end_x - start_x >= display_width and end_y - start_y >= display_height:
            # Crop exactly the size we need for display
            cropped_face = resized_face[start_y:start_y+display_height, start_x:start_x+display_width]
            return cropped_face
        else:
            # Rare case where we still don't have enough pixels
            # Create a buffer and place the image in the center
            buffer = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            # Calculate offsets to center the image
            offset_x = max(0, (display_width - (end_x - start_x)) // 2)
            offset_y = max(0, (display_height - (end_y - start_y)) // 2)
            
            # Get the valid part of the resized face
            valid_height = end_y - start_y
            valid_width = end_x - start_x
            
            # Copy the valid portion of the resized face into the buffer
            buffer[offset_y:offset_y+valid_height, offset_x:offset_x+valid_width] = resized_face[start_y:end_y, start_x:end_x]
            return buffer
    
    # Fallback
    return np.zeros((display_height, display_width, 3), dtype=np.uint8)

class FaceRecognizer:
    def __init__(self, known_faces_path='known_faces.pkl'):
        self.known_faces_path = known_faces_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from pickle file if it exists"""
        try:
            if os.path.exists(self.known_faces_path):
                with open(self.known_faces_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} known faces")
            else:
                print("No known faces file found. Starting with empty database.")
        except Exception as e:
            print(f"Error loading known faces: {e}")
            
    def save_known_faces(self):
        """Save known faces to pickle file"""
        try:
            with open(self.known_faces_path, 'wb') as f:
                data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names,
                    'timestamp': datetime.now().isoformat()
                }
                pickle.dump(data, f)
            print(f"Saved {len(self.known_face_names)} known faces")
        except Exception as e:
            print(f"Error saving known faces: {e}")
            
    def add_face(self, face_image, name):
        """Add a new face to the database"""
        try:
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                print("No face found in the image")
                return False
                
            # Get the encoding for the first face found
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            
            # Check if this face is already known
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                if True in matches:
                    # Face already exists
                    match_index = matches.index(True)
                    print(f"Face already exists as {self.known_face_names[match_index]}")
                    return False
            
            # Add the new face
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Save the updated faces
            self.save_known_faces()
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
            
    def recognize_face(self, face_image):
        """Recognize a face from the database"""
        if not self.known_face_encodings:
            return "Unknown"
            
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                return "No face detected"
                
            # Get encoding for the first face
            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            
            # Calculate face distances for better matching
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
            
            # Always show the best match, but with confidence level indicated
                if matches[best_match_index] and confidence > 0.5:
                    return f"{self.known_face_names[best_match_index]} ({confidence:.2f})"
                elif confidence > 0.4:  # Lower threshold for showing a name
                    return f"{self.known_face_names[best_match_index]} (Low: {confidence:.2f})"
                else:
                    # If we're really unsure, still show the closest match with warning
                    return f"Maybe {self.known_face_names[best_match_index]}? ({confidence:.2f})"
        
            return "Unknown"
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return "Error"
    
    def get_known_face_count(self):
        """Return number of known faces"""
        return len(self.known_face_names)
        
    def get_known_face_names(self):
        """Return list of known face names"""
        return self.known_face_names
        
    def remove_face(self, name):
        """Remove a face from the database by name"""
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            self.save_known_faces()
            return True
        return False

def faceDetection():
    cuda_available = check_cuda()
    
    # Initialize face recognizer
    face_recognizer = FaceRecognizer()
    
    # Use DNN-based face detector
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    # Check if model files exist, if not - download them
    if not (os.path.isfile(model_file) and os.path.isfile(config_file)):
        print("Downloading face detection model files...")
        os.system("curl -O https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        os.system("curl -O https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
        os.rename("res10_300x300_ssd_iter_140000_fp16.caffemodel", model_file)
    
    # Load the DNN face detector using OpenCV
    try:
        net = cv2.dnn.readNet(model_file, config_file)
        print(f"Successfully loaded face detection model from {model_file} and {config_file}")
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        return
    
    # Use CUDA if available
    if cuda_available:
        if torch.cuda.is_available():
            print("Using PyTorch with CUDA acceleration")
        else:
            print("Setting OpenCV DNN backend to CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Open webcam - try multiple indices if first attempt fails
    cap = None
    for i in range(3):  # Try webcam indices 0, 1, 2
        print(f"Trying to open webcam at index {i}")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened webcam at index {i}")
            break
    
    if cap is None or not cap.isOpened():
        print("Failed to open webcam. Make sure it's connected and not used by another application.")
        return
    
    # Create named windows
    cv2.namedWindow('Face Detection (DNN)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detected Face', cv2.WINDOW_NORMAL)
    
    # Variables to control face rectangle
    face_margin_percent = 0  # Size control: 0% means default size
    offset_x = 0  # Horizontal position offset in pixels
    offset_y = 0  # Vertical position offset in pixels
    auto_rotate = True  # Toggle for auto-rotation feature
    
    # Recognition mode and state variables
    recognition_enabled = True
    adding_new_face = False
    deleting_face = False
    new_face_name = ""
    delete_face_input = ""
    face_recognition_cooldown = 0
    
    # Fixed display dimensions for face window
    display_width = 300
    display_height = 300
    
    # Print instructions
    print("Controls:")
    print("- Press '+' or '=' to enlarge face rectangle")
    print("- Press '-' or '_' to shrink face rectangle")
    print("- Press arrow keys or WASD to move face rectangle")
    print("- Press 'o' to toggle auto-rotation")
    print("- Press 'r' to reset position and size")
    print("- Press 'f' to toggle face recognition")
    print("- Press 'n' to add a new face (then type name and press Enter)")
    print("- Press 'x' to delete a face (then type name or number and press Enter)")
    print("- Press 'l' to list all known faces")
    print("- Press 'q' to quit")
    
    
    # Store the last detected rotation angle for smoothing
    last_rotation_angle = 0
    current_face_name = "Unknown"
    
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to capture frame ({frame_count} frames captured so far)")
            if frame_count == 0:
                # Try to fix webcam if no frames were captured
                print("Trying to reinitialize webcam...")
                cap.release()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Could not reinitialize webcam. Exiting.")
                    break
                continue
            break
        
        frame_count += 1
        if frame_count == 1:
            print(f"Successfully captured first frame: {frame.shape}")
        
        # Create a clean copy of the frame for face extraction before any drawing
        clean_frame = frame.copy()
        
        # Prepare frame for face detection
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        except Exception as e:
            print(f"Error creating blob: {e}")
            continue
        
        # Use PyTorch if available
        if torch.cuda.is_available():
            try:
                # Convert blob to PyTorch tensor and move to GPU
                tensor = torch.from_numpy(blob).cuda()
                
                # Convert back to numpy for OpenCV
                blob = tensor.cpu().numpy()
            except Exception as e:
                print(f"Error using PyTorch CUDA: {e}")
        
        try:
            net.setInput(blob)
            detections = net.forward()
        except Exception as e:
            print(f"Error in DNN forward pass: {e}")
            continue
        
        # Process detections
        h, w = frame.shape[:2]
        
        # Prepare a blank image for "no face detected" case
        face_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Find face with highest confidence
        best_confidence = 0
        best_face_coords = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Filter by confidence
                
                # Track the face with highest confidence
                if confidence > best_confidence:
                    best_confidence = confidence
                    
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    best_face_coords = (x1, y1, x2, y2)
                
                # Draw rectangle on main frame (NOT on the clean copy)
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add confidence text
                label = f"{confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the detected face or "No face detected" text
        if best_face_coords:
            x1, y1, x2, y2 = best_face_coords
            
            # Calculate face dimensions
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Apply the margin to adjust rectangle size
            margin_x = int(face_width * face_margin_percent / 100)
            margin_y = int(face_height * face_margin_percent / 100)
            
            # Apply margins and offsets to coordinates
            x1 = max(0, min(w-1, x1 - margin_x + offset_x))
            y1 = max(0, min(h-1, y1 - margin_y + offset_y))
            x2 = max(0, min(w, x2 + margin_x + offset_x))
            y2 = max(0, min(h, y2 + margin_y + offset_y))
            
            # Draw the adjusted rectangle on the frame with a different color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Extract face from the CLEAN frame using adjusted coordinates
            if y1 < y2 and x1 < x2:  # Ensure valid crop region
                face_crop = clean_frame[y1:y2, x1:x2].copy()
                
                # Perform face recognition if enabled
                if recognition_enabled and face_crop.size > 0 and face_recognition_cooldown <= 0:
                    current_face_name = face_recognizer.recognize_face(face_crop)
                    face_recognition_cooldown = 15  # Reset cooldown (frames)
                elif face_recognition_cooldown > 0:
                    face_recognition_cooldown -= 1
                
                rotation_angle = 0
                if auto_rotate and face_crop.size > 0:
                    # Detect face orientation
                    current_angle = detect_face_orientation(face_crop)
                    
                    # Apply temporal smoothing to reduce jitter
                    rotation_angle = 0.7 * last_rotation_angle + 0.3 * current_angle
                    last_rotation_angle = rotation_angle
                    
                    # Show the rotation angle on the main frame
                    cv2.putText(frame, f"Rotation: {rotation_angle:.1f}°", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Process face to make it upright and fill the frame
                    face_display = process_face_upright(
                        face_crop, 
                        rotation_angle,  # Pass the detected angle
                        display_width, 
                        display_height
                    )
                else:
                    # No auto-rotation - just resize to fill
                    face_display = process_face_upright(
                        face_crop, 
                        0,  # No rotation
                        display_width, 
                        display_height
                    )
                
                # Display the recognized name on both frames if recognition is enabled
                if recognition_enabled:
                    cv2.putText(frame, f"Person: {current_face_name}", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                    cv2.putText(face_display, current_face_name, 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
                # Show if we're in "add new face" mode
                if adding_new_face:
                    cv2.putText(frame, f"Adding new face: {new_face_name}_", 
                              (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                elif deleting_face:
                    cv2.putText(frame, f"Delete face: {delete_face_input}_", 
                      (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
                    
            else:
                print(f"Invalid crop region: ({x1}, {y1}) to ({x2}, {y2})")
                
        else:
            # No face detected
            face_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(face_display, "No face detected", (30, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Show the current settings on the main frame
        status_text = f"Size: {face_margin_percent}% | X: {offset_x}px | Y: {offset_y}px | Rotation: {'AUTO' if auto_rotate else 'OFF'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show frames - Make sure they're not empty
        if frame.size > 0:
            cv2.imshow('Face Detection (DNN)', frame)
        if face_display.size > 0:
            cv2.imshow('Detected Face', face_display)
        
        # Check for keyboard input with lower timeout for more responsive UI
        key = cv2.waitKey(1) & 0xFF
        
        # Process key presses - prioritize name input mode
        if adding_new_face:
            # When in name input mode, only handle name input related keys
            # Handle backspace
            if key == 8 or key == 127:  # Backspace
                new_face_name = new_face_name[:-1]
                print(f"Entering name: {new_face_name}_")
            # Handle Enter to confirm
            elif key == 13 or key == 10:  # Enter
                if new_face_name:
                    if face_recognizer.add_face(face_crop, new_face_name):
                        print(f"Added new face: {new_face_name}")
                    else:
                        print("Failed to add face.")
                    adding_new_face = False
                    new_face_name = ""
                else:
                    print("Name cannot be empty.")
            # Handle Escape to cancel name input
            elif key == 27:  # Escape key
                print("Cancelled adding new face")
                adding_new_face = False
                new_face_name = ""
            # Handle printable characters
            elif 32 <= key <= 126:  # Printable ASCII
                new_face_name += chr(key)
                print(f"Entering name: {new_face_name}_")
            # Ignore all other keys when in name input mode
        elif deleting_face:
            if key == 8 or key == 127: # backspace
                delete_face_input = delete_face_input[:-1]
                print(f"Enter a name/number to delete: {delete_face_input}_")
                
            elif key == 13 or key == 10: # enter
                if delete_face_input:
                    try:
                        idx = int(delete_face_input) - 1
                        known_faces = face_recognizer.get_known_face_names()
                        if 0 <= idx < len(known_faces):
                            name_to_delete = known_faces[idx]
                            if face_recognizer.remove_face(name_to_delete):
                                print(f"Deleted face: {name_to_delete}")
                            else:
                                print(f"Failed to delete face: {name_to_delete}")
                        else:
                            print(f"Invalid index.")
                    except ValueError:
                        if face_recognizer.remove_face(delete_face_input):
                            print(f"Deleted face: {delete_face_input}")
                        else:
                            print(f"Face not found: {delete_face_input}")
                    
                    deleting_face = False
                    delete_face_input = ""
                    
                    known_faces = face_recognizer.get_known_face_names()
                    print(f"Known faces are now ({len(known_faces)})")
                    for i, name in enumerate(known_faces):
                        print(f"{i+1}. {name}")
                        
                else:
                    print("Name or number cannot be empty.")
                    
            elif key == 27: # esc
                print("Cancelled deleting face")
                deleting_face = False
                delete_face_input = ""
            elif 32 <= key <= 126:
                delete_face_input += chr(key)
                print(f"Enter a name or number to delete {delete_face_input}_")
        else:
            # Only process these keys when NOT in name input mode
            if key == ord('q'):
                break
            
            # Size control keys
            elif key == ord('+') or key == ord('='):
                face_margin_percent += 5
                print(f"Increased face margin to {face_margin_percent}%")
            elif key == ord('-') or key == ord('_'):
                face_margin_percent -= 5
                print(f"Decreased face margin to {face_margin_percent}%")
            
            # Position control keys - multiple options for cross-platform compatibility
            # Left: Left arrow or 'a'
            elif key == ord('a') or key == 81 or key == 2 or key == 65361:
                offset_x -= 5
                print(f"Moved left: offset_x = {offset_x}")
            # Right: Right arrow or 'd' 
            elif key == ord('d') or key == 83 or key == 3 or key == 65363:
                offset_x += 5
                print(f"Moved right: offset_x = {offset_x}")
            # Up: Up arrow or 'w'
            elif key == ord('w') or key == 82 or key == 0 or key == 65362:
                offset_y -= 5
                print(f"Moved up: offset_y = {offset_y}")
            # Down: Down arrow or 's'
            elif key == ord('s') or key == 84 or key == 1 or key == 65364:
                offset_y += 5
                print(f"Moved down: offset_y = {offset_y}")
            
            # Toggle auto-rotation
            elif key == ord('o'):
                auto_rotate = not auto_rotate
                print(f"Auto-rotation: {'ON' if auto_rotate else 'OFF'}")
            
            # Reset position
            elif key == ord('r'):
                offset_x = 0
                offset_y = 0
                face_margin_percent = 0
                print("Reset face position and size")
                
            # Toggle face recognition
            elif key == ord('f'):
                recognition_enabled = not recognition_enabled
                print(f"Face recognition: {'ON' if recognition_enabled else 'OFF'}")
                
            # Add new face mode
            elif key == ord('n'):
                if best_face_coords and frame_count > 0:
                    adding_new_face = True
                    new_face_name = ""
                    print("Adding new face. Type name and press Enter:")
                    print("(Press ESC to cancel)")
                else:
                    print("No face detected to add.")
            
            # List known faces
            elif key == ord('l'):
                known_faces = face_recognizer.get_known_face_names()
                print(f"Known faces ({len(known_faces)}):")
                for i, name in enumerate(known_faces):
                    print(f"{i+1}. {name}")
            
            elif key == ord('x'):
                known_faces = face_recognizer.get_known_face_names()
                if known_faces:
                    deleting_face = True
                    delete_face_input = ""
                    print("Deleting face. Type name or number of face and press Enter:")
                    print("Press Esc to cancel")
                    
                    print(f"Known faces ({len(known_faces)}):")
                    
                    for i, name in enumerate(known_faces):
                        print(f"{i+1}. {name}")
                else:
                    print("No faces in database to delete.")
                
    # Clean up
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    faceDetection()
    cv2.destroyAllWindows()
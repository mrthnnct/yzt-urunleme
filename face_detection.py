import cv2
import numpy as np
import os
import sys
import torch
import math
import mediapipe as mp

## mediapipe is now implemented correctly

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

def faceDetection():
    cuda_available = check_cuda()
    
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
    net = cv2.dnn.readNet(model_file, config_file)
    
    # Use CUDA if available
    if cuda_available:
        if torch.cuda.is_available():
            print("Using PyTorch with CUDA acceleration")
        else:
            print("Setting OpenCV DNN backend to CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    cap = cv2.VideoCapture(0)
    
    # Create named windows
    cv2.namedWindow('Face Detection (DNN)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detected Face', cv2.WINDOW_NORMAL)
    
    # Variables to control face rectangle
    face_margin_percent = 0  # Size control: 0% means default size
    offset_x = 0  # Horizontal position offset in pixels
    offset_y = 0  # Vertical position offset in pixels
    auto_rotate = True  # Toggle for auto-rotation feature
    
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
    print("- Press 'q' to quit")
    
    # Store the last detected rotation angle for smoothing
    last_rotation_angle = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame")
            break
            
        # Create a clean copy of the frame for face extraction before any drawing
        clean_frame = frame.copy()
        
        # Prepare frame for face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        
        # Use PyTorch if available
        if torch.cuda.is_available():
            # Convert blob to PyTorch tensor and move to GPU
            tensor = torch.from_numpy(blob).cuda()
            
            # Convert back to numpy for OpenCV
            blob = tensor.cpu().numpy()
        
        net.setInput(blob)
        detections = net.forward()
        
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
            face_crop = clean_frame[y1:y2, x1:x2].copy()
            
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
        else:
            # No face detected
            face_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(face_display, "No face detected", (30, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show the current settings on the main frame
        status_text = f"Size: {face_margin_percent}% | X: {offset_x}px | Y: {offset_y}px | Rotation: {'AUTO' if auto_rotate else 'OFF'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show frames
        cv2.imshow('Face Detection (DNN)', frame)
        cv2.imshow('Detected Face', face_display)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Process key presses
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
    
    # Clean up
    cap.release()
    face_mesh.close()

if __name__ == "__main__":
    faceDetection()
    cv2.destroyAllWindows()
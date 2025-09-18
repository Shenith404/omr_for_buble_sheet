
import numpy as np
from ultralytics import YOLO



def detect_digits(image, model_path=r"F:\University\fyp\mcq_test_1\destopApp\models\reg_detection\best.pt", show_plots=True):
  
    try:
        # Load YOLO model
        yolo_model = YOLO(model_path)
        
    
        if image is None:
            return {"success": False, "error": "Could not load image. Check the file path."}
       
        # Run YOLO detection
        results = yolo_model(image)
    
        # Get detection results
        detections = results[0]
        boxes = detections.boxes
        
        result_data = {
            "success": True,
            "total_detections": 0,
            "detected_digits": [],
            "digit_sequence": "",
            "complete_number": "",
            "validation_messages": [],
            "confidence_assessment": ""
        }
        
        if boxes is not None and len(boxes) > 0:
            result_data["total_detections"] = len(boxes)
            
            # Process each detected digit
            detected_digits = []
        
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name (digit) from the model
                class_name = yolo_model.names[class_id]
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                
                
                detected_digits.append({
                    'digit': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'x_center': x_center,
                    'y_center': y_center
                })
        
            # Validation 1: Check center point positions (x-coordinate should not be less than 200)
            low_position_digits = [d for d in detected_digits if d['x_center'] < 200]
            if low_position_digits:
                result_data["success"]= False
                result_data["validation_messages"].append(
                    f"Some digits are detected too far left in the image (x < 200). Please ensure the image is properly aligned."
                )

            # Sort detections by x-coordinate (left to right)
            detected_digits.sort(key=lambda x: x['x_center'])
            result_data["detected_digits"] = detected_digits
            
            # Validation 2: Check if total digits equals 8
            if len(detected_digits) != 8:
                result_data["success"]= False
                result_data["validation_messages"].append(
                    f"Expected 8 digits, but detected {len(detected_digits)}. Please ensure the registration number is fully visible."
                )

            
            # Validation 3: Confidence assessment
            confidences = [d['confidence'] for d in detected_digits]
            avg_confidence = np.mean(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            if avg_confidence >= 0.8:
                confidence_level = "HIGH"
                confidence_comment = f"High confidence detections (avg: {avg_confidence:.3f}). Results are likely accurate."
            elif avg_confidence >= 0.6:
                confidence_level = "MEDIUM"
                confidence_comment = f"Medium confidence detections (avg: {avg_confidence:.3f}). Results are moderately reliable."
            else:
                confidence_level = "LOW"
                confidence_comment = f"Low confidence detections (avg: {avg_confidence:.3f}). Results may be unreliable, consider better image quality."
            
            result_data["confidence_assessment"] = confidence_comment


            
            # Generate final sequence
            digit_sequence = [d['digit'] for d in detected_digits]
            complete_number = ''.join(digit_sequence)
            
            result_data["digit_sequence"] = ' '.join(digit_sequence)
            result_data["complete_number"] = complete_number
            

        
        else:
            result_data["validation_messages"].append("No digits detected in the image")
           
        
        return result_data
        
    except Exception as e:
        return {"success": False, "error": f"Error during processing: {e}"}

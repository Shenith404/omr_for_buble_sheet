import os
import cv2
import numpy as np
from PySide6.QtCore import Signal, QObject
import utils
import model
from . import reg_detection_shaded as reg_detection


class OMRProcessor(QObject):
    """
    Optimized OMR processing worker that handles batch processing of answer sheets
    Features:
    - Thread-safe operation
    - Progress reporting
    - Error handling
    - Memory-efficient processing
    """
    
    # Signals for communication with main thread
    progress_updated = Signal(int, str)  # progress_percent, status_message
    image_processed = Signal(str, list, np.ndarray)  # filename, answers, marked_image
    processing_complete = Signal()
    error_occurred = Signal(str)
    cancelled = Signal()

    def __init__(self, image_paths, project_path, model_answers, model_answers_2=None):
        super().__init__()
        self.image_paths = image_paths
        self.project_path = project_path
        self.model_answers = model_answers  # Model answers for comparison (List 1)
        self.model_answers_2 = model_answers_2  # Model answers for shuffle (List 2)
        self._cancel_requested = False
        self.widthImg = 1025  # Standard OMR sheet width
        self.heightImg = 760  # Standard OMR sheet height
        self.batch_size = 50  # Process images in batches to manage memory
        self.dummy_answer = [0] * 50  # Placeholder for dummy answers
        self.total_marks = 0  # Initialize total marks

    def process_all(self):
        """Process all images with accurate progress tracking"""
        print(self.model_answers)
        print(self.model_answers_2)
        try:
            total_images = len(self.image_paths)
            processed_count = 0
            
            for image_path in self.image_paths:
                if self._cancel_requested:
                    self.cancelled.emit()
                    break
                
                filename = os.path.basename(image_path)
                
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Failed to load image: {filename}")
                    
                    # Process image
                    detected_answers, final_img, new_filename = self.process_omr_sheet(image)
                    
                    # Rename the image if a new filename is generated
                    if new_filename is not None and new_filename != filename:
                        try:
                            new_image_path = os.path.join(os.path.dirname(image_path), new_filename)
                            # Check if target file already exists
                            if os.path.exists(new_image_path):
                                print(f"Warning: Target file {new_filename} already exists. Skipping rename.")
                            else:
                                os.rename(image_path, new_image_path)
                                filename = new_filename
                                image_path = new_image_path
                                print(f"File renamed from {os.path.basename(image_path)} to {new_filename}")
                        except OSError as e:
                            print(f"Error renaming file: {str(e)}")
                            # Continue processing with original filename
                    # Emit progress after completion (not before)
                    processed_count += 1
                    progress = int((processed_count / total_images) * 100)
                    self.progress_updated.emit(
                        progress,
                        f"Processed {filename} ({processed_count}/{total_images})"
                    )
                    
                    # Emit results
                    self.image_processed.emit(filename, detected_answers, final_img)
                    
                except Exception as e:
                    self.error_occurred.emit(f"Skipped {filename}: {str(e)}")
                    continue
            
            if not self._cancel_requested:
                self.processing_complete.emit()
        
        except Exception as e:
            self.error_occurred.emit(f"Fatal processing error: {str(e)}")

    def process_omr_sheet(self, image):
        """
        Process a single OMR sheet with optimized operations
        Returns:
        - answers: List of detected answers (1-based index)
        - marked_image: Image with marked answers
        """
        # Step 1: Preprocessing with optimized operations
        img = cv2.resize(image, (self.widthImg, self.heightImg))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        edges = cv2.Canny(blur, 10, 50)

        #disable  mark button when processing
        
        if self._cancel_requested:
            raise RuntimeError("Processing cancelled")

        # Step 2: Contour Detection with area filtering
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = utils.rectContour(contours)
        
        if not rects:
            return self.dummy_answer, img, None
        

        
        #Step 3: detect registration number contour and rename the file accordingly
        new_filename = None  # Initialize new_filename at the beginning
        
        if len(rects) > 1:  # Check if we have at least 2 rectangles before accessing rects[1]
            reg_contour = utils.getCornerPoints(rects[1]) 
            
            if reg_contour.size != 0:
                reg_contour = utils.reorder(reg_contour)
                pts1r = np.float32(reg_contour)
                pts2r = np.float32([[0, 0], [1000, 0], [0, 100], [1000, 100]])
                matrix = cv2.getPerspectiveTransform(pts1r, pts2r)
                warped_r = cv2.warpPerspective(img, matrix, (1000, 100))
                
                try:
                    reg_number =  reg_detection.detect_reg_number(warped_r)
                    print("Reg Number Detection: ", reg_number)

                    if reg_number["success"]:
                        # Rename the file as registration number
                        reg_num_str = reg_number["digit_sequence"]
                        if reg_num_str:  # Check if digit_sequence is not empty
                            new_filename = f"{reg_num_str}.png"
                        else:
                            print("Warning: Registration number detected but digit sequence is empty")
                    else:
                        print("Warning: Registration number detection failed or no digits detected")
                except Exception as e:
                    print(f"Error during registration number detection: {str(e)}")
            else:
                print("Warning: Registration contour not found or invalid")
        else:
            print("Warning: Insufficient rectangles detected for registration number processing")
               




        # Step 4: Perspective Transform with error checking
        biggest = utils.getCornerPoints(rects[0])

        if biggest.size == 0:
            return self.dummy_answer, img, None
            
        biggest = utils.reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [self.widthImg, 0], [0, self.heightImg], [self.widthImg, self.heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, matrix, (self.widthImg, self.heightImg))

        # Step 5: Adaptive Thresholding for better robustness
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        #increase brightness
        #warped_gray =cv2.convertScaleAbs(warped_gray, alpha=1, beta=50)
       

       # totalPixelSize =cv2.countNonZero(thresh)
       # print("Total Pixel Size: ",totalPixelSize)

        if self._cancel_requested:
            raise RuntimeError("Processing cancelled")
        thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological Opening (remove small white dots)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
        erosion_image =cv2.erode(thresh, kernel_open, iterations=1)

        # Step 1: Split into boxes
        boxes = utils.verticalSplitBoxes(erosion_image)
        #thresh_boxes =utils.verticalSplitBoxes(thresh)
        detected_answers = []

        # Step 2: For each question box
        for i,box in enumerate(boxes):
            if self._cancel_requested:
                raise RuntimeError("Processing cancelled")

            # Get answer bubbles (skip unwanted blocks)
            answer_blocks = utils.getAnswerBlocks(boxes[i])[2:6]  # Adjust if needed
            #thresh_answer_blocks =utils.getAnswerBlocks(thresh_boxes[i])[2:6]
            
            #if total pixel value of thresh_answer_block is greater than 240 replace relevent anwer_block by increasing constrast
            # for j, tab in enumerate(thresh_answer_blocks):
            #     p_val = cv2.countNonZero(tab)
            #     if p_val > 1700:  # Careful: You had a typo (944480), it should be 255*64*64 = 1044480
            #         print("psfds", p_val)
            #         answer_blocks[j] = cv2.convertScaleAbs(answer_blocks[j], alpha=1.5, beta=50)

            # Step 3: Collect non-empty blocks for batch processing
            valid_blocks = [(j, block) for j, block in enumerate(answer_blocks) if cv2.countNonZero(block) > 0]

            if valid_blocks:
                indices, blocks = zip(*valid_blocks)  # unzip indices and images

                # Step 4: Batch classify
                predictions = model.classify_batch(list(blocks))  # returns list of (label, confidence)

                # Step 5: Get crossed bubble index (if only one)
                marked = []
                for idx, (label, _) in zip(indices, predictions):
                    if label == "cross_sheets_adpthresh":
                        marked.append(idx + 2)  # 1-based index since we skipped first 2 blocks

                detected_answers.append(marked[0] if len(marked) == 1 else -1)
            else:
                detected_answers.append(-1)  # No valid bubbles

        # Step 6: Generate Results with optimized drawing
        drawing = np.zeros_like(warped)
        # Select model answers based on number of rectangles detected
        # If len(rects) > 2, use model_answers (row 2), else use model_answers_2 (row 3)
        if len(rects) > 2 and cv2.contourArea(rects[2])>2000:
            
            drawing,self.total_marks = utils.showAnswers(drawing, detected_answers,self.model_answers)

            print(f"Rectangle count: {len(rects)}, Using Model Answer 1 (Row 2)")
        else:
            drawing,self.total_marks = utils.showAnswers(drawing, detected_answers,self.model_answers_2)

            print(f"Rectangle count: {len(rects)}, Using Model Answer 2 (Row 3)")
        
        
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        inv_drawing = cv2.warpPerspective(drawing, inv_matrix, (img.shape[1], img.shape[0]))
        final_img = cv2.addWeighted(img, 1, inv_drawing, 1, 0)
        cv2.putText(final_img, f"Total Marks: {self.total_marks}/50", (50, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 250), 1)

        return detected_answers, final_img,new_filename

    def cancel(self):
        """Request cancellation of current processing"""
        self._cancel_requested = True

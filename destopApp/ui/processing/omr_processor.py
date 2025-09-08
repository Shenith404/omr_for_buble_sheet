import os
import cv2
import numpy as np
from PySide6.QtCore import Signal, QObject
import utils
import model


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

    def __init__(self, image_paths, project_path, model_answers):
        super().__init__()
        self.image_paths = image_paths
        self.project_path = project_path
        self.model_answers = model_answers  # Model answers for comparison
        self._cancel_requested = False
        self.widthImg = 1025  # Standard OMR sheet width
        self.heightImg = 760  # Standard OMR sheet height
        self.batch_size = 50  # Process images in batches to manage memory
        self.dummy_answer = [0] * 50  # Placeholder for dummy answers
        self.total_marks = 0  # Initialize total marks

    def process_all(self):
        """Process all images with accurate progress tracking"""
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
                    detected_answers, final_img = self.process_omr_sheet(image)
                    
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
            return self.dummy_answer, img

        # Step 3: Perspective Transform with error checking
        biggest = utils.getCornerPoints(rects[0])
        if biggest.size == 0:
            return self.dummy_answer, img
            
        biggest = utils.reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [self.widthImg, 0], [0, self.heightImg], [self.widthImg, self.heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, matrix, (self.widthImg, self.heightImg))

        # Step 4: Adaptive Thresholding for better robustness
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
        drawing,self.total_marks = utils.showAnswers(drawing, detected_answers, self.model_answers)
        
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        inv_drawing = cv2.warpPerspective(drawing, inv_matrix, (img.shape[1], img.shape[0]))
        final_img = cv2.addWeighted(img, 1, inv_drawing, 1, 0)
        cv2.putText(final_img, f"Total Marks: {self.total_marks}/50", (50, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 250), 1)

        return detected_answers, final_img

    def cancel(self):
        """Request cancellation of current processing"""
        self._cancel_requested = True

import os
import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


class ImageDisplayHandler:
    """Handles image display and navigation functionality"""
    
    @staticmethod
    def display_original_image(image_path, image_label, lbl_image_info, current_index, total_images):
        """Display the original image without showing marked answers"""
        try:
            filename = image_path.split('\\')[-1]  # Get filename from path

            # Load the original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image")

            # Convert the image from BGR (OpenCV format) to RGB (Qt format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # Create a QImage object from the RGB image data
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Scale the pixmap to fit the QLabel while maintaining the aspect ratio
            scaled_pixmap = pixmap.scaled(
                image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Set the scaled pixmap to the QLabel for display
            image_label.setPixmap(scaled_pixmap)
            lbl_image_info.setText(
                f"Image {current_index + 1}/{total_images}\n"
                f"{filename}"
            )

        except Exception as e:
            # Handle any errors that occur during image loading or display
            print(f"Image display error: {str(e)}")
            image_label.setText(f"Error: {str(e)}")
            lbl_image_info.setText("Image display error")

    @staticmethod
    def display_marked_image(marked_image, image_label, lbl_image_info, current_index, total_images):
        """Display marked image with optimized rendering"""
        try:
            # Convert color space
            rgb_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Create QImage
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale to fit
            scaled_pixmap = pixmap.scaled(
                image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            image_label.setPixmap(scaled_pixmap)
            lbl_image_info.setText(
                f"Processed Image {current_index + 1}/{total_images}\n"
                f"Marked results displayed"
            )
            
        except Exception as e:
            print(f"Marked image display error: {str(e)}")
            image_label.setText(f"Error displaying results")


class ImageNavigationHandler:
    """Handles image navigation functionality"""
    
    @staticmethod
    def update_navigation_buttons(btn_prev, btn_next, btn_process_all, image_paths, current_index):
        """Update button states based on current position"""
        has_images = len(image_paths) > 0
        btn_prev.setEnabled(has_images and current_index > 0)
        btn_next.setEnabled(has_images and current_index < len(image_paths) - 1)
        btn_process_all.setEnabled(has_images)

    @staticmethod
    def validate_image_paths(image_paths):
        """
        Validate image paths input
        Args:
            image_paths: List of paths to images
        """
        # Validate input
        if not isinstance(image_paths, list):
            raise TypeError("image_paths must be a list")
        if not all(isinstance(p, (str, os.PathLike)) for p in image_paths):
            raise TypeError("All paths must be strings or PathLike objects")
        
        return True

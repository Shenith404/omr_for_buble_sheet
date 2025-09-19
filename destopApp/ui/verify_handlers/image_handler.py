"""
Image Handlers for Verify Tab
Handles image loading, display, navigation, and filtering functionality
"""

import os
from PySide6.QtWidgets import QListWidgetItem, QMessageBox
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class VerifyImageHandler:
    """Handles image operations for the Verify Tab"""
    
    @staticmethod
    def load_project_images(project_path):
        """Load processed images from project's results folder"""
        image_paths = []
        
        # Load processed images from results folder
        results_dir = os.path.join(project_path, "results")
        if os.path.exists(results_dir):
            image_paths = [
                os.path.join(results_dir, f)
                for f in sorted(os.listdir(results_dir))
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
        
        return image_paths

    @staticmethod
    def display_current_image(image_path, image_label):
        """Display the current processed image"""
        try:
            # Load and display image
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap.scaled(
                image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            return True
        except Exception as e:
            image_label.setText(f"Error loading image: {str(e)}")
            return False

    @staticmethod
    def apply_search_filter(image_paths, search_text):
        """Apply search filter to image list"""
        if not search_text:
            return image_paths
        
        search_text = search_text.lower()
        return [
            path for path in image_paths
            if search_text in os.path.basename(path).lower()
        ]

    @staticmethod
    def apply_review_status_filter(image_paths, filter_type, is_reviewed_callback):
        """Apply review status filter to image list"""
        if filter_type == "All Images":
            return image_paths
        elif filter_type == "Reviewed Only":
            return [
                path for path in image_paths
                if is_reviewed_callback(os.path.basename(path))
            ]
        elif filter_type == "Unreviewed Only":
            return [
                path for path in image_paths
                if not is_reviewed_callback(os.path.basename(path))
            ]
        return image_paths

    @staticmethod
    def update_image_list_widget(images_list, filtered_image_paths, is_reviewed_callback, is_verified_callback=None):
        """Update the image list widget with filtered results"""
        images_list.clear()
        
        for image_path in filtered_image_paths:
            filename = os.path.basename(image_path)
            is_reviewed = is_reviewed_callback(filename)
            is_verified = is_verified_callback(filename) if is_verified_callback else False
            
            # Create list item with status indicator
            if is_verified:
                status_icon = "✅✅"  # Double check for verified
                status_text = "Verified"
            elif is_reviewed:
                status_icon = "✅"   # Single check for reviewed
                status_text = "Reviewed"
            else:
                status_icon = "⏳"   # Pending
                status_text = "Pending"
                
            item_text = f"{status_icon} {filename}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, image_path)  # Store full path in item data
            
            # Set tooltip
            item.setToolTip(f"{status_text}: {filename}")
            
            images_list.addItem(item)

    @staticmethod
    def update_image_count_info(lbl_image_info, image_paths, filtered_image_paths, current_index):
        """Update the image count information display"""
        total_images = len(image_paths)
        filtered_count = len(filtered_image_paths)
        
        if total_images > 0:
            if filtered_count == total_images:
                count_text = f"{total_images} images"
            else:
                count_text = f"{filtered_count}/{total_images} images"
            
            if image_paths and current_index < len(image_paths):
                current_image_path = image_paths[current_index]
                if current_image_path in filtered_image_paths:
                    filtered_index = filtered_image_paths.index(current_image_path) + 1
                    count_text = f"{filtered_index}/{filtered_count} ({count_text})"
        else:
            count_text = "No images loaded"
            
        lbl_image_info.setText(count_text)

    @staticmethod
    def highlight_current_image_in_list(images_list, current_image_path):
        """Highlight the current image in the list widget"""
        for i in range(images_list.count()):
            item = images_list.item(i)
            item_path = item.data(Qt.UserRole)
            if item_path == current_image_path:
                images_list.setCurrentItem(item)
                break

    @staticmethod
    def get_image_index_from_list_item(item, image_paths):
        """Get the image index from a list widget item"""
        image_path = item.data(Qt.UserRole)
        if image_path in image_paths:
            return image_paths.index(image_path)
        return -1

    @staticmethod
    def validate_image_navigation(current_index, total_images, direction):
        """Validate if navigation in the given direction is possible"""
        if direction == "next":
            return current_index < total_images - 1
        elif direction == "prev":
            return current_index > 0
        return False

    @staticmethod
    def update_navigation_button_states(btn_prev, btn_next, current_index, total_images):
        """Update navigation button enabled/disabled states"""
        has_images = total_images > 0
        btn_prev.setEnabled(has_images and current_index > 0)
        btn_next.setEnabled(has_images and current_index < total_images - 1)

    @staticmethod
    def show_image_loading_error(image_label, error_message):
        """Display an error message when image loading fails"""
        image_label.setText(f"Error loading image: {error_message}")

    @staticmethod
    def show_no_images_message(image_label, message_type="no_project"):
        """Show appropriate message when no images are available"""
        if message_type == "no_project":
            image_label.setText(" No reviewed images to verify\n\nOpen a project from the Project tab to begin")
        elif message_type == "processing_required":
            image_label.setText("Please process the images first using the Processing tab")
        elif message_type == "no_results":
            image_label.setText("No reviewed images available for verification")
        elif message_type == "no_reviewed_images":
            image_label.setText("No reviewed images available for verification\n\nPlease review images in the Review tab first")
        else:
            image_label.setText("No images available")
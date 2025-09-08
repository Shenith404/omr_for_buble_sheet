import os
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QThread
import db


class UIProcessingHandler:
    """Handles UI-related processing operations"""
    
    @staticmethod
    def initialize_processing(project_path, image_paths, model_answers):
        """Initialize processing setup"""
        if not image_paths or not project_path:
            raise Exception("No project or images loaded")
        
        try:
            # Initialize results directory
            results_dir = os.path.join(project_path, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Initialize json file
            handler = db.OMRJsonHandler(project_path)
            # Delete existing json file
            handler.delete_answers_file()
            
            # Save model answers to json file
            handler.save_model_answers(model_answers)
            
            return handler
            
        except Exception as e:
            raise Exception(f"Error starting processing: {str(e)}")

    @staticmethod
    def setup_worker_thread(image_paths, project_path, model_answers, omr_processor_class):
        """Setup worker thread with lower priority"""
        worker_thread = QThread()
        worker_thread.setPriority(QThread.LowPriority)
        omr_processor = omr_processor_class(image_paths, project_path, model_answers)
        omr_processor.moveToThread(worker_thread)
        
        return worker_thread, omr_processor

    @staticmethod
    def update_progress_ui(progress, message, progress_bar, lbl_status):
        """Throttled progress updates"""
        # Only update if progress increased or it's a completion message
        if progress > progress_bar.value() or progress == 100:
            progress_bar.setValue(progress)
            lbl_status.setText(message)
            
            # Process events at certain intervals (every 5% or completion)
            if progress % 5 == 0 or progress == 100:
                QApplication.processEvents()  # Ensure UI remains responsive

    @staticmethod
    def finish_processing_ui(processed_count, total_count, project_path, 
                           progress_bar, lbl_status, btn_process_all, 
                           btn_cancel, btn_delete_image, worker_thread):
        """Verify all images were processed and allow navigation to other tabs"""
        if processed_count < total_count:
            lbl_status.setText(
                f"Completed {processed_count}/{total_count} images"
            )
        else:
            lbl_status.setText("Processing completed successfully!")
        
        progress_bar.setValue(100)
        btn_process_all.setEnabled(True)
        btn_cancel.setEnabled(False)
        btn_delete_image.setEnabled(True)  # Enable delete button

        # Ensure the worker thread is properly terminated
        if hasattr(worker_thread, 'isRunning') and worker_thread.isRunning():
            worker_thread.quit()
            worker_thread.wait()

        QMessageBox.information(
            None,
            "Processing Complete",
            f"Processed {processed_count}/{total_count} images\n"
            f"Results saved to: {os.path.join(project_path, 'results')}"
        )

    @staticmethod
    def cancel_processing_ui(worker_thread, omr_processor, progress_bar, 
                           btn_process_all, btn_cancel):
        """Handle processing cancellation and reset state"""
        if hasattr(omr_processor, 'cancel'):
            omr_processor.cancel()

        # Ensure the worker thread is properly terminated
        if hasattr(worker_thread, 'isRunning') and worker_thread.isRunning():
            worker_thread.quit()
            worker_thread.wait()

        # Reset processing state
        progress_bar.setValue(0)
        btn_process_all.setEnabled(True)
        btn_cancel.setEnabled(False)

    @staticmethod
    def handle_processing_error_ui(error_msg, btn_process_all, btn_cancel, 
                                 lbl_status, progress_bar):
        """Handle processing errors with user feedback"""
        btn_process_all.setEnabled(True)
        btn_cancel.setEnabled(False)
        lbl_status.setText(f"Error: {error_msg}")
        progress_bar.setValue(0)
        
        # Show error message but don't block processing
        QMessageBox.critical(None, "Processing Error", error_msg)


class UIStateHandler:
    """Handles UI state management"""
    
    @staticmethod
    def update_project_ui(project_path, lbl_project, btn_save_model_answers, btn_edit_answers):
        """Update UI after project is loaded"""
        lbl_project.setText(os.path.basename(project_path))
        btn_save_model_answers.setEnabled(True)  # Enable the button when a project is loaded
        btn_edit_answers.setEnabled(True)  # Enable the edit button when a project is loaded

    @staticmethod
    def update_image_info_ui(image_paths, lbl_image_info):
        """Update image information in UI"""
        if image_paths:
            # Image info will be updated in show_current_image
            pass
        else:
            lbl_image_info.setText("No valid images found")

    @staticmethod
    def reset_processing_state():
        """Reset processing state variables"""
        return {
            'processed_images': {},
            'current_answers': {},
            'processed_count': 0
        }

    @staticmethod
    def enable_processing_buttons(btn_process_all, btn_cancel, btn_delete_image, enable_processing=True):
        """Enable/disable processing related buttons"""
        btn_process_all.setEnabled(enable_processing)
        btn_cancel.setEnabled(not enable_processing)
        btn_delete_image.setEnabled(enable_processing)

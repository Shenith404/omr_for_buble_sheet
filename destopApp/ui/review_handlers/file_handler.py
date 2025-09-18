"""
File Management Handlers for Review Tab
Handles file operations like rename, download, and project management
"""

import os
import shutil
from PySide6.QtWidgets import QMessageBox, QFileDialog


class ReviewFileHandler:
    """Handles file operations for the Review Tab"""
    
    @staticmethod
    def rename_image_file(project_path, current_filename, new_filename):
        """Rename image file in all relevant directories"""
        if not new_filename or not current_filename:
            raise ValueError("Invalid filename provided")
        
        # Remove any path separators and invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            new_filename = new_filename.replace(char, '')
        
        if not new_filename:
            raise ValueError("Filename contains only invalid characters")
        
        # Get file extension from current filename
        current_name, current_ext = os.path.splitext(current_filename)
        new_filename_with_ext = new_filename + current_ext
        
        # Check if file already exists
        new_result_path = os.path.join(project_path, "results", new_filename_with_ext)
        if os.path.exists(new_result_path):
            current_result_path = os.path.join(project_path, "results", current_filename)
            if new_result_path != current_result_path:
                raise FileExistsError(f"A file named '{new_filename_with_ext}' already exists")
        
        # Rename files in all relevant directories
        directories = ["results", "original_images"]
        renamed_paths = {}
        
        for directory in directories:
            old_file_path = os.path.join(project_path, directory, current_filename)
            new_file_path = os.path.join(project_path, directory, new_filename_with_ext)
            
            if os.path.exists(old_file_path):
                os.rename(old_file_path, new_file_path)
                renamed_paths[directory] = new_file_path
        
        return new_filename_with_ext, renamed_paths

    @staticmethod
    def update_database_filename(handler, old_filename, new_filename):
        """Update filename in the database"""
        if handler:
            try:
                # Get the sheet data for the old filename
                sheet_data = handler.get_sheet(old_filename)
                if sheet_data:
                    # Update the filename in the database
                    handler.update_filename(old_filename, new_filename)
                    return True
            except Exception as e:
                print(f"Warning: Could not update database entry: {e}")
                return False
        return False

    @staticmethod
    def download_project_results(project_path, download_dir=None):
        """Download all project results to selected directory"""
        if not project_path:
            raise ValueError("No project loaded")
        
        # Let user select download directory if not provided
        if not download_dir:
            download_dir = QFileDialog.getExistingDirectory(
                None,
                "Select Download Directory",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
        
        if not download_dir:
            return None  # User cancelled
        
        # Get the project title using project path
        p_title = project_path.split(os.sep)[-1]
        
        # Create results directory in download location
        dest_dir = os.path.join(download_dir, p_title + "_marked_sheets")
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy all result images
        results_src = os.path.join(project_path, "results")
        if os.path.exists(results_src):
            for file in os.listdir(results_src):
                src_file = os.path.join(results_src, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dest_dir)
        
        # Copy JSON data files
        json_files = [f for f in os.listdir(project_path) if f.endswith('.json')]
        for json_file in json_files:
            src_file = os.path.join(project_path, json_file)
            shutil.copy2(src_file, dest_dir)
        
        return dest_dir

    @staticmethod
    def export_results_to_excel(handler, output_path):
        """Export results to Excel file"""
        if handler:
            try:
                handler.export_to_excel(output_path)
                return True
            except Exception as e:
                print(f"Error exporting to Excel: {e}")
                return False
        return False

    @staticmethod
    def validate_filename(filename):
        """Validate filename for illegal characters"""
        if not filename:
            return False, "Filename cannot be empty"
        
        # Remove any path separators and invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            if char in filename:
                return False, f"Filename contains invalid character: {char}"
        
        # Check for reserved names on Windows
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        if filename.upper() in reserved_names:
            return False, f"'{filename}' is a reserved filename"
        
        return True, "Valid filename"

    @staticmethod
    def get_file_extension(filename):
        """Get the file extension from filename"""
        return os.path.splitext(filename)[1]

    @staticmethod
    def get_filename_without_extension(filename):
        """Get the filename without extension"""
        return os.path.splitext(filename)[0]

    @staticmethod
    def update_image_paths_after_rename(image_paths, old_filename, new_filename):
        """Update image paths list after file rename"""
        updated_paths = []
        for path in image_paths:
            if os.path.basename(path) == old_filename:
                # Replace the filename part with new name
                new_path = os.path.join(os.path.dirname(path), new_filename)
                updated_paths.append(new_path)
            else:
                updated_paths.append(path)
        return updated_paths

    @staticmethod
    def copy_file_safely(src_path, dest_path):
        """Copy file with error handling"""
        try:
            # Create destination directory if it doesn't exist
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy the file
            shutil.copy2(src_path, dest_path)
            return True
        except Exception as e:
            print(f"Error copying file {src_path} to {dest_path}: {e}")
            return False

    @staticmethod
    def ensure_directory_exists(directory_path):
        """Ensure directory exists, create if it doesn't"""
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False

    @staticmethod
    def get_directory_file_count(directory_path, extensions=None):
        """Get count of files in directory with optional extension filter"""
        if not os.path.exists(directory_path):
            return 0
        
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        count = 0
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                count += 1
        
        return count

    @staticmethod
    def clean_filename(filename):
        """Clean filename by removing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        cleaned = filename
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '')
        return cleaned.strip()

    @staticmethod
    def get_safe_filename(base_name, directory, extension):
        """Generate a safe filename that doesn't conflict with existing files"""
        counter = 1
        safe_name = base_name + extension
        
        while os.path.exists(os.path.join(directory, safe_name)):
            safe_name = f"{base_name}_{counter}{extension}"
            counter += 1
        
        return safe_name
import os
import csv
import subprocess
import cv2
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Protection
from openpyxl.worksheet.datavalidation import DataValidation
from PySide6.QtWidgets import QMessageBox
import db


class FileOperationHandler:
    """Handles file operations and project management"""
    
    @staticmethod
    def load_project_images(project_path):
        """Load project with optimized file handling"""
        image_paths = []
        
        try:
            # Load from original_images folder
            images_dir = os.path.join(project_path, "original_images")
            if os.path.exists(images_dir):
                image_paths.extend([
                    os.path.join(images_dir, f) 
                    for f in sorted(os.listdir(images_dir))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ])
            
            # Load linked images from references file
            ref_file = os.path.join(project_path, "image_references.txt")
            if os.path.exists(ref_file):
                with open(ref_file, 'r') as f:
                    image_paths.extend([
                        line.strip() 
                        for line in f 
                        if line.strip() and os.path.exists(line.strip())
                    ])
        
        except Exception as e:
            raise Exception(f"Error loading project: {str(e)}")
        
        return image_paths

    @staticmethod
    def delete_image_file(image_path):
        """Delete image file from disk"""
        try:
            filename = os.path.basename(image_path)
            os.remove(image_path)
            return f"'{filename}' has been deleted."
        except Exception as e:
            raise Exception(f"Failed to delete image: {str(e)}")

    @staticmethod
    def save_image_results(filename, answers, marked_image, total_marks, project_path, handler):
        """Save results immediately after each image is processed"""
        if not project_path or marked_image is None:
            return
            
        results_dir = os.path.join(project_path, "results")
        
        try:
            # 1. Ensure directory exists
            os.makedirs(results_dir, exist_ok=True)
            
            # 2. Save marked image
            output_path = os.path.join(results_dir, f"{filename}")
            cv2.imwrite(output_path, marked_image)
            
            # 3. Save answers in a json file
            handler.create_or_update_sheet(filename, answers, total_marks)
                    
        except Exception as e:
            raise Exception(f"Error saving {filename}: {str(e)}")
        
    @staticmethod
    def clear_results_folder(project_path):
        """Clear all files in the results folder"""
        if not project_path:
            return
        
        results_dir = os.path.join(project_path, "results")
        
        try:
            if os.path.exists(results_dir):
                for f in os.listdir(results_dir):
                    file_path = os.path.join(results_dir, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        except Exception as e:
            raise Exception(f"Error clearing results folder: {str(e)}")


class ModelAnswersHandler:
    """Handles model answers file operations"""
    
    @staticmethod
    def create_model_answers_file(file_path):
        """Create an Excel file where only the second row is editable with allowed values 0-4"""
        wb = Workbook()
        ws = wb.active

        # Write header (Q1 to Q50) in row 1
        for col in range(1, 51):
            cell = ws.cell(row=1, column=col, value=f"Q{col}")
            cell.protection = Protection(locked=True)  # lock header

        # Write default answers (0s) in row 2
        for col in range(1, 51):
            cell = ws.cell(row=2, column=col, value=0)
            cell.protection = Protection(locked=False)  # allow editing

        # Add data validation (0 to 4) for editable answer cells
        dv = DataValidation(type="whole", operator="between", formula1=0, formula2=4)
        dv.error = "Please enter a number between 0 and 4."
        dv.errorTitle = "Invalid Input"
        ws.add_data_validation(dv)
        dv.add("A2:AX2")  # 50 columns = A to AX

        # Lock sheet
        ws.protection.sheet = True
        ws.protection.enable()

        # Save the file
        wb.save(file_path)

    @staticmethod
    def open_file_for_editing(file_path):
        """Open the XLSX file directly for editing"""
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        else:  # macOS or Linux
            subprocess.call(('open' if os.name == 'posix' else 'xdg-open', file_path))

    @staticmethod
    def read_model_answers_file(file_path):
        """Read the model answers from the XLSX file and update the processor"""
        wb = load_workbook(file_path, data_only=True)
        ws = wb.active
        row = [ws.cell(row=2, column=col).value for col in range(1, 51)]
        return [int(ans) if ans is not None else 0 for ans in row]

    @staticmethod
    def save_model_answers_workflow(project_path):
        """Complete workflow for saving model answers"""
        if not project_path:
            raise Exception("No project is open. Please select a project first.")

        # Define the path for the model answers XLSX file
        model_answers_path = os.path.join(project_path, "model_answers.xlsx")

        # Check if the file exists; if not, create it with default answers and a header
        if not os.path.exists(model_answers_path):
            ModelAnswersHandler.create_model_answers_file(model_answers_path)

        # Load the answers from the file after editing
        model_answers = ModelAnswersHandler.read_model_answers_file(model_answers_path)
        for ans in model_answers:
            if ans not in range(1, 5):
                raise ValueError("Model answers must be between 1 and 4.")

        return model_answers

    @staticmethod
    def edit_answers_workflow(project_path):
        """Open the model answers XLSX file for editing"""
        if not project_path:
            raise Exception("No project is open. Please select a project first.")

        model_answers_path = os.path.join(project_path, "model_answers.xlsx")
        if os.path.exists(model_answers_path):
            ModelAnswersHandler.open_file_for_editing(model_answers_path)
        else:
            raise Exception("Model answers file does not exist. Please load it first.")

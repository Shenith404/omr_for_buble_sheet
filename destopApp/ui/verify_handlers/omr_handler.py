"""
OMR Processing Handlers for Verify Tab
Handles OMR-specific operations like answer changing, verification marking, and question management
"""

import os
import cv2
from PySide6.QtWidgets import QMessageBox
import utils


class VerifyOMRHandler:
    """Handles OMR-specific operations for the Verify Tab"""
    
    @staticmethod
    def update_question_button_appearance(question_buttons, handler, current_image_filename):
        """Update question button appearance to show current answers"""
        if not handler or not current_image_filename:
            return
        
        try:
            sheet_data = handler.get_sheet(current_image_filename)
            if sheet_data and 'detected_answers' in sheet_data:
                answers = sheet_data['detected_answers']
                
                for i, btn in enumerate(question_buttons):
                    q_num = i + 1
                    if i < len(answers):
                        answer = answers[i]  # 0-based index
                        
                        # Update button style and tooltip based on answer
                        if answer == -1:  # No answer detected
                            btn.setStyleSheet("""
                                QPushButton {
                                    background: #4a4a4a;
                                    border: 1px solid #666666;
                                    border-radius: 4px;
                                    color: #cccccc;
                                    font-size: 11px;
                                    font-weight: 600;
                                    padding: 0px;
                                    margin: 0px;
                                    min-width: 20px;
                                    min-height: 20px;
                                    max-width: 22px;
                                    max-height: 22px;
                                }
                                QPushButton:hover {
                                    background: #5a5a5a;
                                    border-color: #0078d4;
                                }
                                QPushButton:checked {
                                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #0078d4, stop:1 #005a9e);
                                    border: 2px solid #0078d4;
                                    color: #ffffff;
                                }
                            """)
                            btn.setToolTip(f"Question {q_num}\nNo answer detected")
                            btn.setText(str(q_num))
                        else:
                            # Convert 1-based answer to letter (2->A, 3->B, 4->C, 5->D)
                            answer_letter = chr(ord('A') + answer - 2) if answer >= 2 else '?'
                            btn.setStyleSheet("""
                                QPushButton {
                                    background: #2d5a2d;
                                    border: 1px solid #4a8f4a;
                                    border-radius: 4px;
                                    color: #ffffff;
                                    font-size: 9px;
                                    font-weight: 600;
                                    padding: 0px;
                                    margin: 0px;
                                    min-width: 20px;
                                    min-height: 20px;
                                    max-width: 22px;
                                    max-height: 22px;
                                }
                                QPushButton:hover {
                                    background: #3a6a3a;
                                    border-color: #0078d4;
                                }
                                QPushButton:checked {
                                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #0078d4, stop:1 #005a9e);
                                    border: 2px solid #0078d4;
                                    color: #ffffff;
                                }
                            """)
                            btn.setToolTip(f"Question {q_num}\nDetected: {answer_letter}")
                            # Show answer letter on button
                            btn.setText(f"{q_num}\n{answer_letter}")
                    else:
                        # Reset to default style
                        VerifyOMRHandler.reset_question_button_style(btn, q_num)
        except Exception as e:
            print(f"Error updating question buttons: {e}")
            # Reset all buttons to default on error
            for i, btn in enumerate(question_buttons):
                VerifyOMRHandler.reset_question_button_style(btn, i + 1)

    @staticmethod
    def reset_question_button_style(btn, question_num):
        """Reset question button to default style"""
        btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border: 1px solid #505050;
                border-radius: 4px;
                color: #e8e8e8;
                font-size: 11px;
                font-weight: 600;
                padding: 0px;
                margin: 0px;
                min-width: 20px;
                min-height: 20px;
                max-width: 22px;
                max-height: 22px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:1 #3a3a3a);
                border: 2px solid #0078d4;
                color: #ffffff;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: 2px solid #0078d4;
                color: #ffffff;
            }
        """)
        btn.setText(str(question_num))
        btn.setToolTip(f"Question {question_num}\nClick to select")

    @staticmethod
    def update_question_selection(question_buttons, selected_question_label, question_num):
        """Update question selection state"""
        # Update button states
        for i, btn in enumerate(question_buttons):
            btn.setChecked(i + 1 == question_num)
        
        # Update selected question label
        selected_question_label.setText(f"Selected: Question {question_num}")

    @staticmethod
    def update_answer_selection(answer_buttons, answer_combo, answer_num):
        """Update answer selection state"""
        # Update answer button states
        for i, btn in enumerate(answer_buttons):
            btn.setChecked(i + 1 == answer_num)
        
        # Update the hidden combo box for compatibility
        answer_combo.setCurrentText(str(answer_num))

    @staticmethod
    def change_detected_answer(handler, project_path, current_image_filename, question_num, new_answer, model_answers):
        """Change the detected answer for selected question"""
        if not handler or not current_image_filename:
            raise ValueError("Handler or image filename is missing")
        
        try:
            # Update the answer in the database
            new_detected_answers = handler.update_correction(
                current_image_filename,
                question_num - 1,  # Convert to 0-based index
                new_answer + 1  # Detected answers saved as 1->2 2->3 3->4 4->5 and no answers saved as -1
            )

            # Get original image from original_images folder
            original_image_path = os.path.join(
                project_path, "original_images", current_image_filename
            )
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")
            
            img = cv2.imread(original_image_path)
            
            # Update the image with new answer
            result_img = utils.process_omr_sheet_without_model(
                img,
                new_detected_answers,
                model_answers
            )
            
            # Save the updated image back to results folder
            result_image_path = os.path.join(
                project_path, "results", current_image_filename
            )
            cv2.imwrite(result_image_path, result_img)
            
            return result_image_path
            
        except Exception as e:
            raise Exception(f"Failed to update answer: {str(e)}")

    @staticmethod
    def mark_as_verified(handler, project_path, current_image_filename, reviewed_images_set):
        """Mark current image as verified by second examiner"""
        if not current_image_filename:
            raise ValueError("No image filename provided")
        
        reviewed_images_set.add(current_image_filename)
        
        image_path = os.path.join(project_path, "results", current_image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        try:
            # Draw the verification stamp (different from review stamp)
            verified_img = utils.draw_stamp(img, input_name="Second Examiner", position=(150, 50), color=(0, 255, 0))  # Green stamp
            cv2.imwrite(image_path, verified_img)

            # Update database to mark as verified
            if handler:
                handler.mark_for_verification(current_image_filename, True)
            
            return image_path
            
        except Exception as e:
            raise Exception(f"Error drawing verification stamp: {str(e)}")

    @staticmethod
    def is_image_reviewed(handler, filename, reviewed_images_set):
        """Check if image is reviewed using database or fallback set"""
        if not handler:
            return filename in reviewed_images_set
        
        sheet_data = handler.get_sheet(filename)
        if sheet_data:
            return sheet_data.get("reviewed", False)
        return filename in reviewed_images_set

    @staticmethod
    def is_image_verified(handler, filename, verified_images_set):
        """Check if image is verified by second examiner"""
        if not handler:
            return filename in verified_images_set
        
        sheet_data = handler.get_sheet(filename)
        if sheet_data:
            return sheet_data.get("verified", False)
        return filename in verified_images_set

    @staticmethod
    def get_current_answer_for_question(handler, current_image_filename, question_num):
        """Get the current answer for a specific question"""
        if not handler or not current_image_filename:
            return None
        
        try:
            sheet_data = handler.get_sheet(current_image_filename)
            if sheet_data and 'detected_answers' in sheet_data:
                answers = sheet_data['detected_answers']
                if question_num - 1 < len(answers):  # Convert to 0-based index
                    return answers[question_num - 1]
            return None
        except Exception as e:
            print(f"Error getting current answer: {e}")
            return None

    @staticmethod
    def validate_question_answer_input(question_num, answer_num):
        """Validate question and answer input"""
        if not (1 <= question_num <= 50):
            return False, "Question number must be between 1 and 50"
        
        if not (1 <= answer_num <= 4):
            return False, "Answer must be between 1 and 4 (A, B, C, D)"
        
        return True, "Valid input"

    @staticmethod
    def get_answer_letter(answer_num):
        """Convert answer number to letter (1->A, 2->B, 3->C, 4->D)"""
        if 1 <= answer_num <= 4:
            return chr(ord('A') + answer_num - 1)
        return '?'

    @staticmethod
    def get_answer_number(answer_letter):
        """Convert answer letter to number (A->1, B->2, C->3, D->4)"""
        if answer_letter.upper() in 'ABCD':
            return ord(answer_letter.upper()) - ord('A') + 1
        return 0

    @staticmethod
    def update_verify_button_state(btn_mark_verified, handler, current_image_filename, verified_images_set):
        """Update the Verified by Second Examiner button state"""
        if not current_image_filename:
            btn_mark_verified.setEnabled(False)
            return
            
        is_verified = VerifyOMRHandler.is_image_verified(handler, current_image_filename, verified_images_set)
        
        btn_mark_verified.setEnabled(not is_verified)
        btn_mark_verified.setText(
            "✓ Verified" if is_verified else "✓ Verified by Second Examiner"
        )

    @staticmethod
    def load_model_answers(handler):
        """Load model answers from the database"""
        if handler:
            try:
                return handler.read_model_answers()
            except Exception as e:
                print(f"Error loading model answers: {e}")
                return []
        return []

    @staticmethod
    def get_question_summary(handler, current_image_filename):
        """Get a summary of all questions and answers for current image"""
        if not handler or not current_image_filename:
            return {}
        
        try:
            sheet_data = handler.get_sheet(current_image_filename)
            if sheet_data and 'detected_answers' in sheet_data:
                answers = sheet_data['detected_answers']
                summary = {}
                
                for i, answer in enumerate(answers):
                    question_num = i + 1
                    if answer == -1:
                        summary[question_num] = "No Answer"
                    else:
                        # Convert to letter
                        answer_letter = chr(ord('A') + answer - 2) if answer >= 2 else '?'
                        summary[question_num] = f"{answer_letter} ({answer-1})" if answer >= 2 else "Invalid"
                
                return summary
        except Exception as e:
            print(f"Error getting question summary: {e}")
        
        return {}

    @staticmethod
    def count_answered_questions(handler, current_image_filename):
        """Count how many questions have answers detected"""
        if not handler or not current_image_filename:
            return 0, 0  # answered, total
        
        try:
            sheet_data = handler.get_sheet(current_image_filename)
            if sheet_data and 'detected_answers' in sheet_data:
                answers = sheet_data['detected_answers']
                answered = sum(1 for answer in answers if answer != -1)
                total = len(answers)
                return answered, total
        except Exception as e:
            print(f"Error counting answers: {e}")
        
        return 0, 0
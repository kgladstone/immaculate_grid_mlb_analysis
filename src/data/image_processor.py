import os
import sqlite3
import shutil
import pytesseract
import regex as re
import json
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import pandas as pd
import pillow_heif

from utils.constants import (
    GRID_PLAYERS, 
    MY_NAME, 
    LOGO_DARK_PATH, 
    LOGO_LIGHT_PATH, 
    APPLE_TEXTS_DB_PATH,
    IMAGES_PARSER_PATH
)
from utils.utils import ImmaculateGridUtils

# Global Tesseract config for OCR
OCR_CONFIG = r"--oem 3 --psm 6"

class ImageProcessor():
    def __init__(self, db_path, cache_path, image_directory):
        self.db_path = db_path
        self.cache_path = cache_path
        self.image_directory = image_directory

    def _fetch_images(self):
        """
        Query the attachments database
        """
        
        # Map phone numbers to player names
        phone_to_player = {details["phone_number"]: player for player, details in GRID_PLAYERS.items()}
        phone_numbers = list(phone_to_player.keys())

        # Connect to the Messages database
        conn = sqlite3.connect(os.path.expanduser(APPLE_TEXTS_DB_PATH))
        cursor = conn.cursor()

        # Query to find attachments with metadata
        query = f"""
        SELECT 
            attachment.filename, 
            attachment.mime_type, 
            message.date, 
            COALESCE(handle.id, ?) AS sender
        FROM message
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        JOIN message_attachment_join ON message.ROWID = message_attachment_join.message_id
        JOIN attachment ON message_attachment_join.attachment_id = attachment.ROWID
        WHERE (handle.id IN ({','.join(['?' for _ in phone_numbers])})
            OR message.is_from_me = 1)
        AND (attachment.mime_type LIKE 'image/png'
            OR attachment.mime_type LIKE 'image/jpeg'
            OR attachment.mime_type LIKE 'image/jpg'
            OR attachment.mime_type LIKE 'image/heic')
        ORDER BY message.date DESC
        """
            
        try:
            cursor = conn.cursor()
            cursor.execute(query, [MY_NAME] + phone_numbers)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            raise ValueError(f"Database error: {e}")
        finally:
            conn.close()

        cleaned_results = []

        for filename, mime_type, message_date, sender_phone in results:
            if filename:
                path = filename.replace(APPLE_TEXTS_DB_PATH, "")
                submitter = MY_NAME if sender_phone == MY_NAME else phone_to_player.get(sender_phone, "Unknown")
                image_date = ImmaculateGridUtils._convert_timestamp(message_date)
                cleaned_results.append({
                    "path": path, 
                    "mime_type": mime_type,
                    "submitter": submitter, 
                    "image_date": image_date
                    })

        return pd.DataFrame(cleaned_results)
    

    def _validate_images(self, data):
        """
        Validate image metadata to ensure all required fields are present.
        """
        print(f"Validating {len(data)} image metadata entries...")
        if data.empty or not {"path", "submitter", "image_date"}.issubset(data.columns):
            raise ValueError("Validation failed: Missing required fields in image metadata.")
        
    def load_parser_metadata(self):
        """
        Load the parser metadata from the cache file.
        """
        if os.path.exists(IMAGES_PARSER_PATH):
            with open(IMAGES_PARSER_PATH, 'r') as f:
                try:
                    data = json.load(f)
                    # Ensure data is a list of dictionaries
                    if isinstance(data, list) and all(isinstance(entry, dict) for entry in data):
                        return pd.DataFrame(data)
                    else:
                        print("Warning: Parser metadata is not in the expected format (list of dictionaries). Returning an empty DataFrame.")
                        return pd.DataFrame()
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Returning an empty DataFrame.")
                    return pd.DataFrame()
        else:
            print("Parser cache file not found. Returning an empty DataFrame.")
            return pd.DataFrame()
        
    
    def load_image_metadata(self):
        """
        Load the image metadata from the cache file.
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Ensure data is a list of dictionaries
                    if isinstance(data, list) and all(isinstance(entry, dict) for entry in data):
                        return pd.DataFrame(data)
                    else:
                        print("Warning: Metadata is not in the expected format (list of dictionaries). Returning an empty DataFrame.")
                        return pd.DataFrame()
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Returning an empty DataFrame.")
                    return pd.DataFrame()
        else:
            print("Cache file not found. Returning an empty DataFrame.")
            return pd.DataFrame()
        

    def process_images(self):
        """
        Refresh the image folder by copying screenshots from Messages to a specified folder.
        Only process messages from a person/date combination not already in the metadata file.
        """

        # Ensure the save folder exists
        os.makedirs(self.image_directory, exist_ok=True)

        existing_metadata = self.load_image_metadata()

        # Extract existing grid_number and submitter combos
        existing_combinations = {(row.grid_number, row.submitter) for row in existing_metadata.itertuples()}

        # Query the attachments database
        results = self._fetch_images()

        # Collect parser data if parser data file is not empty
        if os.path.exists(IMAGES_PARSER_PATH):
            
            parser_existing_data = self.load_parser_metadata()

            if len(parser_existing_data) > 0:
                # Keep any parser messages with "Invalid image" from the parser_existing_data
                parser_existing_data = parser_existing_data[
                    parser_existing_data["parser_message"].str.contains(
                        "Invalid image",
                        na=False
                    )
                ]

                # Extract paths from parser_existing_data
                existing_paths = set(parser_existing_data["path"])

            else:
                existing_paths = set()

        # Process each attachment
        for _, result in results.iterrows():
            path = os.path.expanduser(result['path'])
            submitter = result['submitter']
            image_date = result['image_date']
            mime_type = result['mime_type']
            parser_message = None

            if path:
                print("*" * 50)
                if os.path.exists(path):

                    if path in existing_paths:
                        print(f"Warning: Skipping existing path {path}")
                        continue

                    print(f"Processing: {path} from {submitter} on {image_date}")

                    # Extract the grid number for checking
                    text = self.extract_text_from_image(path) # OCR operation
                    grid_number = self.grid_number_from_image_text(text) # Text operation

                    if grid_number is None:
                        parser_message = f"Warning: Invalid image. Could not extract grid number from {path}"

                    # Check if this combination already exists in metadata
                    elif (grid_number, submitter) in existing_combinations:
                        parser_message = f"Warning: This grid already exists in metadata (#{grid_number} for {submitter})"

                    # Process the image
                    else:
                        parser_message = self.process_image_with_dynamic_grid(path, submitter, image_date) # OCR operation
            
            parser_data_entry = {
                "path": path,
                "submitter": submitter,
                "image_date": image_date,
                "parser_message": parser_message
            }

            # Append the parser data entry
            print(parser_message)
            self.save_parser_metadata(parser_data_entry)

        print(f"Screenshots saved to {self.image_directory}")


    def safe_cv2_read(self, image_path):
        """
        Safely read an image using OpenCV, handling HEIC format if needed.
        """
        mime_type = self.get_mime_type(image_path)
        if mime_type == 'image/heic':
            return self.read_heic_with_cv2(image_path)
        else:
            return cv2.imread(image_path)
        if img is None:
            print("Error: Unable to load the main image.")
            return None

    
    def read_image(self, file_path):
        """
        Read an image file using PIL, handling HEIC format if needed.
        """

        mime_type = self.get_mime_type(file_path)

        if mime_type == 'image/heic':
            # Read the HEIC file
            heif_file = pillow_heif.open_heif(file_path)
            
            # Convert HEIC data to a PIL Image
            image = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data,
                "raw", heif_file.mode, heif_file.stride
            )
            
            return image  # Return the PIL Image object for further processing
        
        else:
            return Image.open(file_path)


    def read_heic_with_cv2(self, filepath):
        """
        Read a HEIC file using pillow-heif and convert it to a NumPy array compatible with OpenCV.
        """
        # Load the HEIC file using pillow-heif
        heif_file = pillow_heif.open_heif(filepath)
        
        # Convert the HEIC file into a PIL Image
        pil_image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data,
            "raw", heif_file.mode, heif_file.stride
        )
        
        # Convert the PIL Image into a NumPy array (compatible with OpenCV)
        cv2_image = np.array(pil_image)
        
        # Convert RGB to BGR if needed for OpenCV consistency
        if cv2_image.ndim == 3:  # Check if the image has color channels
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        
        return cv2_image

    def crop_text_box_dynamic(self, image_path):
        """
        Detect and crop the text box in the given image dynamically.
        Args:
            image_path (str): Path to the input image.
        Returns:
            PIL.Image: Cropped text box as a PIL Image object, or None if no text box is detected.
        """
        # Load the image
        img = self.safe_cv2_read(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to highlight text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Identify the largest rectangle likely containing the text
        text_box = None
        max_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > max_area and w > h and w > 50:  # Ensure it looks like a text box
                max_area = area
                text_box = (x, y, x + w, y + h)

        if text_box is None:
            print("Error: Unable to detect text box.")
            return None

        # Crop the image to the text box
        x1, y1, x2, y2 = text_box
        cropped_img = img[y1:y2, x1:x2]

        # Convert the cropped image to PIL format
        cropped_pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        return cropped_pil_img


    def convert_cv2_to_pil(self, cv2_image):
        """
        Convert an OpenCV image to a PIL Image.
        """
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


    def preprocess_image_for_logo_finder(self, gray_image):
        """Preprocess the grayscale image to enhance contrast and reduce noise."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced


    def find_logo_position(self, main_image_path, threshold=0.38, scales=None):
        """
        Find the position of the logo in the main image using template matching with scaling.

        Args:
            main_image_path (str): Path to the main image.
            threshold (float): Confidence threshold for template matching.
            scales (list): List of scales to try for the logo image.

        Returns:
            tuple: (top_left, bottom_right) coordinates of the detected logo and the maximum confidence.
        """
        if scales is None:
            scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]

        # Load the main image
        main_image = self.safe_cv2_read(main_image_path)

        # crop the main_image to be the top half and left half
        main_image = main_image[:main_image.shape[0]//2, :main_image.shape[1]//2]

        logo_image_paths = [LOGO_LIGHT_PATH, LOGO_DARK_PATH]

        best_match = None
        best_val = 0

        for i, logo_image_path in enumerate(logo_image_paths):
            logo_image = self.safe_cv2_read(logo_image_path)

            if main_image is None or logo_image is None:
                print("Error: Unable to load one of the images.")
                return None

            # Convert images to grayscale
            main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
            logo_gray = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)

            # Preprocess the grayscale images
            main_preprocessed = self.preprocess_image_for_logo_finder(main_gray)
            logo_preprocessed = self.preprocess_image_for_logo_finder(logo_gray)

            # Iterate over scales
            for scale in scales:
                resized_logo = cv2.resize(logo_preprocessed, (0, 0), fx=scale, fy=scale)

                # Template matching
                result = cv2.matchTemplate(main_preprocessed, resized_logo, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)


                # Update the best match if the confidence is higher
                if max_val > best_val:
                    best_val = max_val
                    best_match = {
                        "top_left": max_loc,
                        "bottom_right": (max_loc[0] + resized_logo.shape[1], max_loc[1] + resized_logo.shape[0]),
                        "confidence": max_val,
                        "scale": scale
                    }

                # print(f"Scale {scale:.2f}, Max confidence: {max_val:.2f}")

        if best_match and best_match["confidence"] >= threshold:
            top_left = best_match["top_left"]
            bottom_right = best_match["bottom_right"]

            # Draw rectangle on the main image
            annotated_image = main_image.copy()
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

            print(f"Success: Logo found at scale {best_match['scale']:.2f}. Confidence: {best_match['confidence']:.2f}")
            
            return top_left, bottom_right, best_match["confidence"]
        else:
            print("Error: Logo not found at any scale.")
            # Draw image
            # img = self.convert_cv2_to_pil(main_image)
            # img.show()
            # quit()
            return None


    def divide_image_into_grid(self, image_path, top_left, bottom_right):
        """
        Divide the main image into a 4x4 grid based on the size of the logo.

        Args:
            image_path (str): Path to the main image.
            top_left (tuple): (x, y) coordinates of the top-left corner of the logo.
            bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the logo.

        Returns:
            list: List of dictionaries containing top-left and bottom-right coordinates for each cell.
        """
        # Load the main image using PIL
        img = self.read_image(image_path)
        img_width, img_height = img.size

        # Assert that the bottom_right second value of the logo is less than half of the image height
        if bottom_right[1] > img_height / 2:
            # draw image and quit
            print("Error: Logo is too low in the image.")
            # # draw logo outline too
            # from PIL import ImageDraw
            # draw = ImageDraw.Draw(img)
            # draw.rectangle([top_left, bottom_right], outline="red")
            # img.show()
            return None

        # Adjust bottom_right so that it caps at 1/4 of the image width
        bottom_right = (int(min(bottom_right[0], img_width/4)), bottom_right[1])

        # Calculate the logo dimensions
        logo_width = bottom_right[0] - top_left[0]
        logo_height = bottom_right[1] - top_left[1]

        # Scalars for cell size adjustments
        WIDTH_SCALAR = img_width / logo_width / 4 * 0.98
        HEIGHT_SCALAR = img_width / logo_width / 4 * 1.10
        TOP_ADJUSTMENT = 0.25
        LEFT_ADJUSTMENT = 0.0015 * logo_width

        # Calculate cell width and height
        cell_width = int(logo_width * WIDTH_SCALAR)
        cell_height = int(logo_height * HEIGHT_SCALAR)

        # Start grid at the top of the logo
        grid_start_x = max(0, top_left[0] - int(LEFT_ADJUSTMENT * cell_width))
        grid_start_y = max(0, top_left[1] - int(TOP_ADJUSTMENT * cell_height))

        grid_cells = []
        

        # Divide the image into a 4x4 grid based on the logo dimensions
        for row in range(4):
            for col in range(4):
                # Calculate the coordinates of the current cell
                x_start = grid_start_x + col * cell_width
                x_end = x_start + cell_width
                y_start = grid_start_y + row * cell_height
                y_end = y_start + cell_height

                # Ensure we don't exceed the image bounds
                if x_end > img_width:
                    x_end = img_width
                if y_end > img_height:
                    y_end = img_height
                
                # Append the cell's coordinates to the list
                if row > 0 and col > 0:
                    grid_cells.append({
                        "row": row - 1,
                        "col": col - 1,
                        "top_left": (x_start, y_start),
                        "bottom_right": (x_end, y_end)
                    })

        # # Display the image with green outlines around each cell
        # from PIL import ImageDraw
        # img_with_grid = img.copy()
        # draw = ImageDraw.Draw(img_with_grid)
        # for cell in grid_cells:
        #     draw.rectangle([cell["top_left"], cell["bottom_right"]], outline="green")
        # img_with_grid.show()

        return grid_cells



    def preprocess_image(self, img, crop_function):
        """
        Preprocess a PIL image to enhance text recognition with extreme contrast enhancement,
        and apply dynamic cropping using a specified crop function.

        Args:
            img (PIL.Image): The input image as a PIL Image object.
            crop_function (function): A function that dynamically crops the image.

        Returns:
            PIL.Image: The preprocessed image after cropping and enhancements.
        """
        # Convert to grayscale
        img = img.convert("L")  # Grayscale

        # Resize for better OCR performance
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)

        # Enhance contrast using autocontrast
        img = ImageOps.autocontrast(img)

        # Apply an additional contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Adjust the factor as needed

        # Apply a binary threshold to make text stand out
        threshold = 128
        img = img.point(lambda p: 255 if p > threshold else 0)

        # Convert the image to OpenCV format for cropping
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)

        # Apply the crop function
        cropped_img = img_cv2
        cropped_img = crop_function(img_cv2)

        if cropped_img is not None and cropped_img.size != 0:
            # Check if cropped image has valid dimensions
            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                # Convert back to PIL Image format after cropping
                img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            else:
                print("Cropped image has invalid dimensions.")
        else:
            print("Cropped image is None or empty.")

        return img


    def dynamic_crop_function(self, image_cv2):
        """
        Detect and crop the text box from the lower half of the image.
        Top and bottom borders are determined based on rows of pixels 
        that are dark for 90% of the row.

        Args:
            image_cv2 (np.ndarray): The input image in OpenCV format.

        Returns:
            np.ndarray: Cropped image as a numpy array, or None if no text box is detected.
        """
        # Get the dimensions of the image
        height, width = image_cv2.shape[:2]

        # Step 1: Crop the lower half of the image
        lower_half = image_cv2[height // 2 :, :]

        # Convert the lower half to grayscale
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)

        # Step 2: Identify the top and bottom borders of the text box
        threshold = 30  # Threshold for "dark" pixel
        dark_ratio = 0.9  # 90% of the row should be dark

        # Iterate through rows to find the top border
        top_border = None
        for i, row in enumerate(gray):
            dark_pixels = np.sum(row < threshold)
            if dark_pixels >= dark_ratio * width:
                top_border = i
                break

        # Iterate through rows (from bottom up) to find the bottom border
        bottom_border = None
        for i in range(gray.shape[0] - 1, -1, -1):
            dark_pixels = np.sum(gray[i] < threshold)
            if dark_pixels >= dark_ratio * width:
                bottom_border = i
                break

        # If borders are not detected, return None
        if top_border is None or bottom_border is None:
            return None

        # Step 3: Crop the image to the detected text box
        cropped_img = lower_half[top_border:bottom_border, :]

        return cropped_img


    def get_mime_type(self, filepath):
        """
        Get the MIME type of a file based on its extension.
        """

        filepath = str(filepath)

        # Define a simple mapping of file extensions to MIME types
        mime_types = {
            r'\.jpg$': 'image/jpeg',
            r'\.jpeg$': 'image/jpeg',
            r'\.png$': 'image/png',
            r'\.gif$': 'image/gif',
            r'\.heic$': 'image/heic',
            r'\.mp4$': 'video/mp4',
            r'\.pdf$': 'application/pdf',
            r'\.txt$': 'text/plain',
        }

        # Match the file extension using regex
        for pattern, mime_type in mime_types.items():
            if re.search(pattern, filepath, re.IGNORECASE):
                return mime_type

        # Return a default MIME type if no match is found
        return 'application/octet-stream'


    def extract_text_from_image(self, image_path):
        """
        Extract text from an image using pytesseract.
        """

        try:
            img = self.read_image(image_path)
            return pytesseract.image_to_string(img, config=OCR_CONFIG)
        except Exception as e:
            return f"Error extracting text: {e}"
        

    def extract_text_from_cells(self, image_path, grid_cells):
        """
        Extract OCR text from each cell in the grid.

        Args:
            image_path (str): Path to the main image.
            grid_cells (list): List of dictionaries with cell coordinates (top-left, bottom-right).

        Returns:
            dict: A dictionary mapping each cell's (row, col) to its OCR text.
        """
        # Load the image
        img = self.safe_cv2_read(image_path)

        cell_texts = {}

        # Process each cell
        for cell in grid_cells:
            row = cell["row"]
            col = cell["col"]
            top_left = cell["top_left"]
            bottom_right = cell["bottom_right"]

            # Crop the cell region from the image
            cropped_cell = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Convert the cropped cell to a PIL image
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cell, cv2.COLOR_BGR2RGB))
            cropped_pil = self.preprocess_image(cropped_pil, self.dynamic_crop_function)  # Apply preprocessing

            # Perform OCR on the cell
            ocr_text = pytesseract.image_to_string(cropped_pil, config=OCR_CONFIG).strip()

            # Save the OCR text to the dictionary
            cell_texts[(row, col)] = self.mlb_player_from_text(ocr_text)

        return cell_texts


    def grid_number_from_image_text(self, text):
        """
        Extract the grid number from the OCR-extracted text.
        """
        match = re.search(r"BASEBALL GRID #(\d+)", text)
        return int(match.group(1)) if match else None


    def mlb_player_from_text(self, text):
        """
        Extract MLB player names from OCR text using regex and filtering.
        Exclude lines containing any reserved words.

        Args:
            text (str): OCR-extracted text.

        Returns:
            str: Concatenated MLB player names.
        """
        reserved_words = ["Previous", "Baseball", "Message", "Soccer", "Football", "Basketball", "Summary"]

        # Helper function for filtering lines
        def filter_reserved_words(lines):
            return [
                line for line in lines
                if not any(word.lower() in line.lower() for word in reserved_words) and len(line) > 2
            ]

        # Split and filter text
        lines = text.splitlines()
        filtered_lines = filter_reserved_words(lines)

        # Use regex to match potential player names in filtered lines
        name_pattern = r"\b[\p{L}\p{M}][\p{L}\p{M}']+\s[\p{L}\p{M}][\p{L}\p{M}']+(?:\s[\p{L}\p{M}][\p{L}\p{M}']+)?\b"
        filtered_text = "\n".join(filtered_lines)
        potential_names = re.findall(name_pattern, filtered_text)

        # Filter names with 2-3 words
        filtered_names = [
            name for name in potential_names
            if len(name.split()) in [2, 3]
        ]

        result = ", ".join(filtered_names)

        # Concatenate player names into a single string
        return result


    def save_consolidated_metadata(self, metadata_new_entry):
        """
        Save or update the consolidated metadata_new_entry JSON file, ensuring no duplicate entries
        for the same grid_number and submitter combination.
        """
        # Load existing metadata_new_entry if the file exists
        consolidated_metadata = self.load_image_metadata()

        # Check for existing data with the same grid_number and submitter
        existing_entry = next(
            (entry for entry in consolidated_metadata.itertuples() if
            entry.grid_number == metadata_new_entry["grid_number"] and
            entry.submitter == metadata_new_entry["submitter"]),
            None
        )

        if existing_entry:
            print(f"Metadata for grid_number {metadata_new_entry['grid_number']} and submitter {metadata_new_entry['submitter']} already exists.")
            return

        # Append the new metadata_new_entry if no duplicate found
        consolidated_metadata = pd.concat(
            [consolidated_metadata, pd.DataFrame([metadata_new_entry])],
            ignore_index=True
        )
        
        # Convert DataFrame to a list of dictionaries
        data_as_dicts = consolidated_metadata.to_dict(orient="records")

        # Write JSON with indentation
        with open(self.cache_path, "w") as f:
            json.dump(data_as_dicts, f, indent=4)

        print(f"Consolidated metadata saved: {self.cache_path}")

    # Save parser metadata
    def save_parser_metadata(self, parser_data_new_entry):
        """
        Save or update the parser metadata JSON file.
        """
        # Load existing metadata if the file exists
        parser_metadata = self.load_parser_metadata()

        # Append the new metadata if no duplicate found
        parser_metadata = pd.concat(
            [parser_metadata, pd.DataFrame([parser_data_new_entry])],
            ignore_index=True
        )

        # drop duplicates
        parser_metadata = parser_metadata.drop_duplicates(subset=["path"], keep="last")

        # Convert DataFrame to a list of dictionaries
        data_as_dicts = parser_metadata.to_dict(orient="records")

        # Write JSON with indentation
        with open(IMAGES_PARSER_PATH, "w") as f:
            json.dump(data_as_dicts, f, indent=4)

        print(f"Parser metadata saved: {IMAGES_PARSER_PATH}")


    def process_image_with_dynamic_grid(self, image_path, submitter, image_date):
        """
        Process an image with dynamic header/footer detection and grid cell OCR assignment.

        Args:
            image_path (str): Path to the input image.
            submitter (str): Name of the submitter.
            image_date (str): Date of the image.

        Returns:
            int: Grid number if successfully processed, None otherwise.
        """

        # Step 1: Find the logo position
        position = self.find_logo_position(image_path)
        if position is None:
            parser_message = f"Warning: Failed to find logo position in {image_path}"
            return parser_message

        # Step 2: Get grid cells based on logo position
        grid_cells = self.divide_image_into_grid(image_path, position[0], position[1])
        if not grid_cells:
            parser_message = f"Warning: Failed to divide grid cells in {image_path}"
            return parser_message

        # Step 3: Extract OCR results from each cell
        players_by_cell = self.extract_text_from_cells(image_path, grid_cells)

        # # Show any subimages that did not extract text
        # for cell, text in players_by_cell.items():
        #     # if text is None:
        #         for cell_data in grid_cells:
        #             if cell_data["row"] == cell[0] and cell_data["col"] == cell[1]:
        #                 top_left = cell_data["top_left"]
        #                 bottom_right = cell_data["bottom_right"]
        #                 cell_image = Image.open(image_path).crop((*top_left, *bottom_right))
        #                 prep = preprocess_image(cell_image, dynamic_crop_function)
        #                 prep.show()

        # Convert players_by_cell into a dictionary indexed by position
        responses = {}
        position_mapping = {
            (0, 0): "top_left",
            (0, 1): "top_center",
            (0, 2): "top_right",
            (1, 0): "middle_left",
            (1, 1): "middle_center",
            (1, 2): "middle_right",
            (2, 0): "bottom_left",
            (2, 1): "bottom_center",
            (2, 2): "bottom_right",
        }

        for (row, col), text in players_by_cell.items():
            position = position_mapping.get((row, col))
            if position:
                responses[position] = text

        # Step 4: Extract grid number from the image
        text = self.extract_text_from_image(image_path)
        grid_number = self.grid_number_from_image_text(text)
        if grid_number is None:
            parser_message = f"Failed to extract grid number from {image_path}"
            return parser_message

        # Step 5: Save the processed image and metadata
        dest_filename = f"{submitter}_{image_date}_grid_{grid_number}.jpg"
        dest_image_path = os.path.join(self.image_directory, dest_filename)
        shutil.copy(image_path, dest_image_path)
        print(f"Copied: {image_path} -> {dest_image_path}")

        metadata = {
            "grid_number": grid_number,
            "submitter": submitter,
            "date": image_date,
            "responses": responses,  # Responses indexed by position
            "image_filename": os.path.basename(dest_image_path),
        }

        print(metadata)

        self.save_consolidated_metadata(metadata)

        parser_message = f"Success: Processed image with grid number {grid_number}"

        return parser_message
import os
import sqlite3
import shutil
from pathlib import Path
import pytesseract
import regex as re
import json
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import cv2
import numpy as np
import pandas as pd
import pillow_heif
import numpy as np
import pandas as pd


from utils.constants import (
    GRID_PLAYERS, 
    MY_NAME, 
    LOGO_DARK_PATH, 
    LOGO_LIGHT_PATH, 
    APPLE_TEXTS_DB_PATH,
    APPLE_IMAGES_PATH,
    IMAGES_PARSER_PATH,
    MESSAGES_CSV_PATH,
)
from utils.utils import ImmaculateGridUtils
from data.messages_loader import MessagesLoader
from data.data_prep import (
    matrix_string_to_flat_list, 
    compare_flat_matrix_with_flat_image_responses
)

class ImageProcessor():
    def __init__(self, db_path, cache_path, image_directory):
        self.db_path = os.path.expanduser(db_path)
        self.cache_path = cache_path
        self.image_directory = image_directory
        self.attachments_source_root = Path(APPLE_IMAGES_PATH).expanduser()
        snapshot_cache = Path.cwd() / "chat_snapshot" / "Attachments"
        # Keep attachments alongside the DB snapshot if possible so the app can read them without FDA
        if snapshot_cache.parent.exists():
            self.attachments_cache_root = snapshot_cache
        else:
            self.attachments_cache_root = Path(self.db_path).parent / "Attachments"
        self._attachments_index = None

        print("*"*20)
        print("Loading images...")
        print("*"*20)

    def _fetch_images(self):
        """
        Query the attachments database
        """
        
        # Map phone numbers to player names
        phone_to_player = dict()
        for player, details in GRID_PLAYERS.items():
            if "phone_number" in details:
                for phone_number in details["phone_number"]:
                    if phone_number not in phone_to_player:
                        phone_to_player[phone_number] = player
        phone_numbers = list(phone_to_player.keys())

        # Connect to the Messages database
        expanded_path = os.path.expanduser(self.db_path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Messages database not found at {expanded_path}")
        if not os.access(expanded_path, os.R_OK):
            raise PermissionError(
                f"Messages database not readable at {expanded_path}. "
                "Grant Full Disk Access to your terminal/IDE or provide a readable copy (e.g., chat_snapshot/chat_backup.db)."
            )

        conn = sqlite3.connect(expanded_path)
        cursor = conn.cursor()

        # Query to find attachments with metadata
        query = f"""
        SELECT 
            attachment.filename,
            attachment.transfer_name,
            attachment.mime_type, 
            message.date, 
            CASE
                WHEN message.is_from_me = 1 THEN ?
                ELSE COALESCE(handle.id, 'Unknown') 
            END AS sender
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

        for filename, transfer_name, mime_type, message_date, sender_phone in results:
            path = None
            if filename:
                path = self._resolve_attachment_path(filename)
            elif transfer_name:
                path = self._find_attachment_by_name(transfer_name)
            if not path:
                continue
            submitter = MY_NAME if sender_phone == MY_NAME else phone_to_player.get(sender_phone, "Unknown")
            image_date = ImmaculateGridUtils._convert_timestamp(message_date)
            cleaned_results.append({
                "path": path,
                "mime_type": mime_type,
                "submitter": submitter,
                "image_date": image_date
                })

        return pd.DataFrame(
            cleaned_results,
            columns=["path", "mime_type", "submitter", "image_date"],
        )

    def _build_attachments_index(self) -> dict:
        if self._attachments_index is not None:
            return self._attachments_index
        index: dict[str, list[str]] = {}
        roots = []
        if self.attachments_cache_root and self.attachments_cache_root.exists():
            roots.append(self.attachments_cache_root)
        if self.attachments_source_root and self.attachments_source_root.exists():
            roots.append(self.attachments_source_root)
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                key = path.name.lower()
                index.setdefault(key, []).append(str(path))
        self._attachments_index = index
        return index

    def _find_attachment_by_name(self, transfer_name: str) -> str | None:
        if not transfer_name:
            return None
        index = self._build_attachments_index()
        matches = index.get(transfer_name.lower())
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        matches.sort(key=lambda p: (os.path.getmtime(p), p), reverse=True)
        return matches[0]
    

    def _resolve_attachment_path(self, raw_path: str) -> str:
        """
        Resolve an attachment path to something readable, falling back to a local cache.
        If the source is unreadable (macOS FDA), encourage syncing Attachments to chat_snapshot.
        """
        expanded = Path(os.path.expanduser(raw_path))

        # Already readable
        if expanded.exists() and os.access(expanded, os.R_OK):
            return str(expanded)

        rel = None
        try:
            rel = expanded.relative_to(self.attachments_source_root)
        except Exception:
            # If the path isn't under the source root, at least preserve the filename
            rel = expanded.name if expanded.name else Path(raw_path).name

        cache_root = self.attachments_cache_root
        if cache_root and rel:
            candidate = cache_root / rel
            if candidate.exists() and os.access(candidate, os.R_OK):
                return str(candidate)

            source_candidate = self.attachments_source_root / rel
            if source_candidate.exists() and os.access(source_candidate, os.R_OK):
                candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_candidate, candidate)
                return str(candidate)

        raise FileNotFoundError(
            f"Attachment not accessible: {expanded}. "
            f"Tried cache at {cache_root}. "
            "Run copy_chat_db.py to sync Attachments or grant Full Disk Access."
        )

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
                        df = pd.DataFrame(data)
                        # Remove duplicates based on all columns
                        df = df.drop_duplicates()
                        return df
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
                        if len(data) > 0:
                            return pd.DataFrame(data).sort_values(by='grid_number')
                        else:
                            return pd.DataFrame()
                    else:
                        print("Warning: Metadata is not in the expected format (list of dictionaries). Returning an empty DataFrame.")
                        return pd.DataFrame()
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Returning an empty DataFrame.")
                    return pd.DataFrame()
        else:
            print("Cache file not found. Returning an empty DataFrame.")
            return pd.DataFrame()
        

    def process_images(self, image_dates_to_parse=None, date_range=None, progress_callback=None):
        """
        Refresh the image folder by copying screenshots from Messages to a specified folder.
        Only process messages from a person/date combination not already in the metadata file.
        """

        # Ensure the save folder exists
        os.makedirs(self.image_directory, exist_ok=True)

        # Query the attachments database
        results = self._fetch_images()

        # Normalize dates and optionally filter the attachments set by date
        results["image_date"] = pd.to_datetime(results["image_date"]).dt.date
        if date_range:
            start_date, end_date = date_range
            if start_date:
                start_date = pd.to_datetime(start_date).date()
                results = results[results["image_date"] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date).date()
                results = results[results["image_date"] <= end_date]
            print(f"Filtered images by date. Remaining: {len(results)}")
        # Convert back to string for serialization/logging downstream
        results["image_date"] = results["image_date"].astype(str)

        # Collect parser data if parser data file is not empty
        if os.path.exists(IMAGES_PARSER_PATH):
            
            parser_existing_data = self.load_parser_metadata()

            if len(parser_existing_data) > 0:
                # Keep any parser messages with "Invalid image" from the parser_existing_data
                parser_existing_data = parser_existing_data[
                    parser_existing_data["parser_message"].str.contains(
                        "Invalid image|already exists|Success|Warning: Grid image is invalid|Failed to",
                        na=False
                    )
                ]

                # Extract paths from parser_existing_data
                existing_paths = set(parser_existing_data["path"])

            else:
                existing_paths = set()

        # Preload messages
        messages_data = MessagesLoader(self.db_path, MESSAGES_CSV_PATH).load().get_data()

        print("**" * 50)
        print("Starting Image Processing...")
        print(f"Image dates to parse: {image_dates_to_parse}")
        print("**" * 50)

        # Sort by date descending to process newest first
        results = results.sort_values(by="image_date", ascending=False)
        total = len(results)
        # Process each attachment
        for idx, (_, result) in enumerate(results.iterrows(), start=1):
            path = self._resolve_attachment_path(result['path'])
            submitter = result['submitter']
            image_date = result['image_date']
            mime_type = result['mime_type']
            parser_message = None

            # # Short circuit by providing image dates to parse
            # if image_dates_to_parse == False:
            #     continue
            # elif isinstance(image_dates_to_parse, list):
            #     if image_date not in image_dates_to_parse:
            #         continue
            # else:
            #     pass

            if path:
                if os.path.exists(path):

                    if path in existing_paths:
                        print(f"Warning: Skipping existing path {path}")
                        # continue

                    print("*" * 50)
                    print(f"Processing: {path} from {submitter} on {image_date}")

                    # Extract the grid number for checking
                    text = self.extract_text_from_image(path) # OCR operation
                    grid_number = self.grid_number_from_image_text(text) # Text operation

                    # Check if the grid number is valid, if not, skip the image and log a message in the parser
                    if grid_number is None:
                        parser_message = f"Warning: Invalid image. Could not extract grid number from {path}"
                        # if "Immaculate Grid Baseball." in path:
                        #     print(path)
                        #     print("Quitting since it should be pulling that one in ^^")
                        #     input("Press Enter to continue...")

                    else:
                        # Extract existing grid_number and submitter combos
                        existing_metadata = self.load_image_metadata()
                        existing_combinations = {(row.grid_number, row.submitter) for row in existing_metadata.itertuples()}

                        # Check if this combination already exists in metadata
                        if (grid_number, submitter) in existing_combinations:
                            parser_message = f"Warning: This grid already exists in metadata (#{grid_number} for {submitter})"
                            print(parser_message)
                            continue

                        # Process the image
                        else:
                            try:
                                matrix = messages_data[(messages_data['grid_number'] == grid_number) & (messages_data['name'] == submitter)]['matrix'].iloc[0]
                                parser_message = self.process_image_with_dynamic_grid(path, submitter, image_date, grid_number, matrix) # OCR operation
                            
                            # There is no corresponding text matrix for this image
                            except IndexError as e:
                                print(e)
                                parser_message = f"Warning: Issue with the text matrix"
            parser_data_entry = {
                "path": path,
                "submitter": submitter,
                "image_date": image_date,
                "parser_message": parser_message
            }

            # Append the parser data entry
            print(parser_message)
            self.save_parser_metadata(parser_data_entry)

            if progress_callback:
                try:
                    progress_callback(idx, total, current_date=image_date, current_submitter=submitter)
                except Exception:
                    # Best-effort progress updates; don't crash processing
                    pass

        print(f"Screenshots saved to {self.image_directory}")


    def safe_cv2_read(self, image_path):
        """
        Safely read an image using OpenCV, handling HEIC format if needed.
        """
        if not image_path:
            return None
        image_path = str(image_path)
        mime_type = self.get_mime_type(image_path)
        if mime_type == 'image/heic':
            return self.read_heic_with_cv2(image_path)
        else:
            return cv2.imread(image_path)

    
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
    

    def get_metadata_from_submitter(self, submitter):
        """
        Return subset of metadata from submitter
        """
        images_metadata = self.load_image_metadata()

        return images_metadata[images_metadata['submitter'] == submitter]
    

    def get_save_reason(self, submitter, grid_number, position):
        images_metadata_submitter = self.get_metadata_from_submitter_grid_numbers(submitter, [grid_number])

        if len(images_metadata_submitter) == 0:
            return "<No Image Data>"
        else:
            responses = images_metadata_submitter['responses'].iloc[0]
            target_response = responses[position]
            if len(target_response) == 0:
                return "<Empty Guess>"
            else:
                return target_response
        

    def get_grid_numbers_from_guess(self, submitter, guess):
        """
        Return all grid numbers for a submitter that contain the guess
        """

        grid_numbers = []

        images_metadata_submitter = self.get_metadata_from_submitter(submitter)

        for _, entry in images_metadata_submitter.iterrows():
            responses = entry['responses']
            grid_number = entry['grid_number']
            for _, response in responses.items():
                if guess in response:
                    grid_numbers.append(grid_number)
        
        return grid_numbers
    

    def get_metadata_from_submitter_grid_numbers(self, submitter, grid_number_list):
        """
        Return subset of metadata based on grid numbers and submitter
        """

        images_metadata_submitter = self.get_metadata_from_submitter(submitter)

        return images_metadata_submitter[images_metadata_submitter['grid_number'].isin(grid_number_list)]
        

    def clear_all_processed_images(self):
        # Load the parser metadata
        parser_metadata = self.load_parser_metadata()

        # Load the images metadata
        images_metadata = self.load_image_metadata()

        # Check initial counts for debugging
        print(f"Initial images metadata count: {len(images_metadata)}")
        print(f"Initial parser metadata count: {len(parser_metadata)}")

        # Remove the entry from the images metadata
        images_metadata = pd.DataFrame()

        # Remove the entry from the parser metadata
        parser_metadata = parser_metadata[
            parser_metadata["parser_message"].str.contains("Warning: Invalid image", na=False)
        ]

        # Check updated counts for debugging
        print(f"Updated images metadata count: {len(images_metadata)}")
        print(f"Updated parser metadata count: {len(parser_metadata)}")

        # Save the updated parser metadata
        parser_metadata_dicts = parser_metadata.to_dict(orient="records")
        with open(IMAGES_PARSER_PATH, "w") as f:
            json.dump(parser_metadata_dicts, f, indent=4)

        # Save the updated images metadata
        images_metadata_dicts = images_metadata.to_dict(orient="records")
        with open(self.cache_path, "w") as f:
            json.dump(images_metadata_dicts, f, indent=4)

        print(f"Cache reset for all processed images.")

    # Function given a submitter and a grid_number, removes it from both the parser metadata and images_metadata
    def remove_entry(self, submitter, grid_number):
        """
        Remove entries matching the submitter and grid_number from the parser and image metadata.
        """
        # Load the parser metadata
        parser_metadata = self.load_parser_metadata()

        # Load the images metadata
        images_metadata = self.load_image_metadata()

        # Check initial counts for debugging
        print(f"Initial images metadata count: {len(images_metadata)}")
        print(f"Initial parser metadata count: {len(parser_metadata)}")

        # Remove the entry from the images metadata
        images_metadata = images_metadata[
            ~((images_metadata["submitter"] == submitter) & (images_metadata["grid_number"] == grid_number))
        ]

        # Remove the entry from the parser metadata
        parser_metadata = parser_metadata[
            ~((parser_metadata["submitter"] == submitter) & (parser_metadata["parser_message"].str.contains("grid number " + str(grid_number), na=False)))
        ]

        # Remove the entry from the parser metadata
        parser_metadata = parser_metadata[
            ~((parser_metadata["submitter"] == submitter) & (parser_metadata["parser_message"].str.contains("#" + str(grid_number), na=False)))
        ]

        # Check updated counts for debugging
        print(f"Updated images metadata count: {len(images_metadata)}")
        print(f"Updated parser metadata count: {len(parser_metadata)}")

        # Save the updated parser metadata
        parser_metadata_dicts = parser_metadata.to_dict(orient="records")
        with open(IMAGES_PARSER_PATH, "w") as f:
            json.dump(parser_metadata_dicts, f, indent=4)

        # Save the updated images metadata
        images_metadata_dicts = images_metadata.to_dict(orient="records")
        with open(self.cache_path, "w") as f:
            json.dump(images_metadata_dicts, f, indent=4)

        print(f"Entry for {submitter} and grid number {grid_number} removed from metadata.")


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
        

    def find_logo_position(self, main_image_path, threshold=0.5, scales=None):
        """
        Find the position of the logo in the main image using template matching with scaling.
        The detected logo position is returned relative to the original image.
        """
        if scales is None:
            scales = [0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        # Load the main image
        main_image = self.safe_cv2_read(main_image_path)
        if main_image is None:
            print("Error: Unable to load the main image.")
            return None

        original_height, original_width = main_image.shape[:2]  # Dimensions of the original image
        vertical_offset = 0
        horizontal_offset = 0

        # Search for the word "Football" using pytesseract
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        ocr_text = pytesseract.image_to_data(main_gray, output_type=pytesseract.Output.DICT)

        for i, text in enumerate(ocr_text["text"]):
            if text.lower() == "football":
                word_top = ocr_text["top"][i]
                word_height = ocr_text["height"][i]
                vertical_offset = word_top + word_height
                main_image = main_image[vertical_offset:, :]  # Crop below the word "Football"
                print(f"Word 'Football' found. Cropping below at y={vertical_offset}.")
                break

        # Crop to the top-left quadrant
        cropped_height = main_image.shape[0] // 2
        cropped_width = main_image.shape[1] // 2
        main_image = main_image[:cropped_height, :cropped_width]
        print(f"Cropped to top-left quadrant: height={cropped_height}, width={cropped_width}.")
        
        # Update offsets for quadrant cropping
        horizontal_offset += 0
        vertical_offset += 0

        logo_image_paths = [LOGO_LIGHT_PATH, LOGO_DARK_PATH]
        best_match = None
        best_val = 0

        for i, logo_image_path in enumerate(logo_image_paths):
            logo_image = self.safe_cv2_read(logo_image_path)
            if logo_image is None:
                print(f"Error: Unable to load the logo image from {logo_image_path}.")
                continue

            # Convert images to grayscale
            cropped_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
            logo_gray = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)

            # Preprocess the grayscale images
            cropped_preprocessed = self.preprocess_image_for_logo_finder(cropped_gray)
            logo_preprocessed = self.preprocess_image_for_logo_finder(logo_gray)

            # Iterate over scales
            for scale in scales:
                try:
                    resized_logo = cv2.resize(logo_preprocessed, (0, 0), fx=scale, fy=scale)

                    # Template matching
                    result = cv2.matchTemplate(cropped_preprocessed, resized_logo, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    # Update the best match if the confidence is higher
                    if max_val > best_val:
                        best_val = max_val
                        best_match = {
                            "top_left": max_loc,
                            "bottom_right": (max_loc[0] + resized_logo.shape[1], max_loc[1] + resized_logo.shape[0]),
                            "confidence": max_val,
                            "scale": scale,
                            "mode": "light" if i == 0 else "dark"
                        }

                except cv2.error as e:
                    print(f"Error at scale {scale:.2f}: {e}")
                    continue


        if best_match and best_match["confidence"] >= threshold:
            # Adjust coordinates to the original image
            top_left = (best_match["top_left"][0] + horizontal_offset, best_match["top_left"][1] + vertical_offset)
            bottom_right = (best_match["bottom_right"][0] + horizontal_offset, best_match["bottom_right"][1] + vertical_offset)

            # Ensure logo is at least 20 pixels wide
            logo_width = bottom_right[0] - top_left[0]
            if logo_width < 20:
                print("Logo found but too small...")
                return None

            print(f"Logo found. Adjusted position to original image: Top-left={top_left}, Bottom-right={bottom_right}.")
            return top_left, bottom_right, best_match["mode"]
        else:
            print("Logo not found at any scale.")
            return None



    def divide_image_into_grid(self, image_path, top_left, bottom_right, cell_scalar=1.0, logo_type="light"):
        """
        Divide the main image into a 4x4 grid based on the size of the logo.

        Args:
            image_path (str): Path to the main image.
            top_left (tuple): (x, y) coordinates of the top-left corner of the logo.
            bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the logo.
            logo_type: "light or "dark"

        Returns:
            list: List of dictionaries containing top-left and bottom-right coordinates for each cell.
        """

        # Load the main image using PIL
        img = self.read_image(image_path)
        img_width, img_height = img.size

        # Calculate the logo width for scaling purposes
        print(bottom_right)
        print(top_left)
        logo_width = (bottom_right[0] - top_left[0])

        # These scale factors are EXTREMELY sensitive
        if logo_type == "light": # assume mobile
            print(f"Logo width: {logo_width}")
            if logo_width < 100:
                DENOMINATOR = 410
            else:
                DENOMINATOR = 382
            cell_width = int(logo_width * (638 / DENOMINATOR) * cell_scalar)
            x_offset = logo_width * (436 / DENOMINATOR)
            y_offset = logo_width * (396 / DENOMINATOR)
        else:
            print(f"Logo width: {logo_width}")
            if logo_width < 105:
                DENOMINATOR = 200
            elif logo_width < 110:
                DENOMINATOR = 195
            elif logo_width < 125:
                DENOMINATOR = 180
            elif logo_width < 170:
                DENOMINATOR = 190
            else:
                DENOMINATOR = 180
            cell_width = int(logo_width * (290 / DENOMINATOR) * cell_scalar)
            x_offset = logo_width * (210 / DENOMINATOR)
            y_offset = logo_width * (180 / DENOMINATOR)

        # Start grid using the offsets
        grid_start_x = top_left[0] + int(x_offset)
        grid_start_y = top_left[1] + int(y_offset)

        grid_cells = []

        # Divide the image into a 3x3 grid based on the logo dimensions
        for row in range(3):
            for col in range(3):
                # Calculate the coordinates of the current cell
                x_start = grid_start_x + col * cell_width
                x_end = x_start + cell_width
                y_start = grid_start_y + row * cell_width
                y_end = y_start + cell_width

                # Ensure we don't exceed the image bounds
                if x_end > img_width:
                    x_end = img_width
                if y_end > img_height:
                    y_end = img_height

                # Ensure grid has a width and height
                if x_end <= x_start or y_end <= y_start:
                    print("Error: Did not parse grid correctly")
                    continue
                
                # Append the cell's coordinates to the list
                grid_cells.append({
                    "row": row,
                    "col": col,
                    "top_left": (int(x_start), int(y_start)),
                    "bottom_right": (int(x_end), int(y_end))
                })
        
        return grid_cells


    def draw_image_with_outlined_logo_and_grid_cells(self, image_path, logo_position, grid_cells, cell_scalar=1):
        img = self.read_image(image_path)
        logo_position = self.find_logo_position(image_path)
        top_left = logo_position[0]
        bottom_right = logo_position[1]
        logo_mode = logo_position[2]
        grid_cells = self.divide_image_into_grid(image_path, top_left, bottom_right, cell_scalar, logo_mode)
        img_with_grid = img.copy()
        draw = ImageDraw.Draw(img_with_grid)
        draw.rectangle([logo_position[0], logo_position[1]], outline="red", width=3)
        for cell in grid_cells:
            draw.rectangle([cell["top_left"], cell["bottom_right"]], outline="green", width=3)
        img_with_grid.show()
        return


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
            return None

        return img


    def dynamic_crop_function(self, image_cv2):
        """
        Detect and crop the text box from the lower portion of the image, ensuring:
        - The top row of the text box is fully dark.
        - The bottom row of the text box is fully dark.
        - Rows between the borders do not have light colors touching the left or right edges.
        - The text box height is less than 1/5 of the original image height.
        - The resulting cropped text box is color-inverted (text is black, background is white).

        Args:
            image_cv2 (np.ndarray): The input image in OpenCV format.

        Returns:
            np.ndarray: Cropped and inverted image containing the text box, or None if no valid text box is detected.
        """

        # Get dimensions of the image
        height, width = image_cv2.shape[:2]

        # Convert the image to grayscale for processing
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        # Step 0: Trivially close out if the whole cell looks uniform
        # Flatten the grayscale image to count pixel occurrences
        unique, counts = np.unique(gray, return_counts=True)

        # Calculate the fraction of pixels for the most common color
        max_fraction = np.max(counts) / gray.size

        # Check if the fraction exceeds the threshold
        if max_fraction >= .90:
            print("Image is too uniform. Returning None.")
            return None

        # Define thresholds
        dark_threshold = 30  # Dark pixel threshold (0-255)
        pct_coverage = .9
        height_adjustment = 5
        PADDING = 0.02  # 2% padding

        # Focus on the lower portion of the image (e.g., lower 1/5)
        lower_portion = gray[-(height // height_adjustment):, :]

        # Get the new height of the cropped lower portion
        lower_height = lower_portion.shape[0]

        # Step 1: Find the top border of the text box
        top_border = None
        for i, row in enumerate(lower_portion):
            dark_pixels = np.sum(row < dark_threshold)
            if dark_pixels >= pct_coverage * width:  # At least pct_coverage% of the row is dark
                top_border = i
                break

        # If no top border is found, return None
        if top_border is None:
            return None

        # Step 2: Find the bottom border of the text box
        bottom_border = None
        for i in range(lower_height - 1, -1, -1):  # Start from the bottom and move upward
            dark_pixels = np.sum(lower_portion[i] < dark_threshold)
            if dark_pixels >= pct_coverage * width:  # At least pct_coverage% of the row is dark
                bottom_border = i
                break

        # If no bottom border is found or if it is above the top border, return None
        if bottom_border is None or bottom_border <= top_border:
            return None


        # Ensure the text box height is less than 1/5 of the original image height
        if (bottom_border - top_border) > height // height_adjustment:
            return None

        # Step 3: Crop the detected text box
        cropped_text_box = lower_portion[top_border:bottom_border + 1, :]

        # Step 4: Add optional padding (adjust based on your needs)
        left_padding = int(PADDING * width)
        right_padding = width - left_padding
        cropped_with_padding = cropped_text_box[:, left_padding:right_padding]

        # Step 5: Invert colors (text to black, background to white)
        inverted_image = cv2.bitwise_not(cropped_with_padding)

        return inverted_image


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
            return pytesseract.image_to_string(img, config=r"--oem 3 --psm 6")
        except Exception as e:
            return f"Error extracting text: {e}"
        

    def count_non_letter_characters(self, input_string):
        """
        Count the number of non-letter characters in a string.

        Args:
            input_string (str): The input string to analyze.

        Returns:
            int: The count of non-letter characters.
        """
        # Use regex to find all characters that are not letters (a-z, A-Z)
        non_letter_characters = re.findall(r'[^a-zA-Z ]', input_string)
        return len(non_letter_characters)
    

    def split_and_get_valid_word(self, input_string):
        """
        Splits a string on '\n' and returns the first part that is a valid word.

        A valid word starts with a capital letter and contains at least one lowercase letter.

        Args:
            input_string (str): The input string.

        Returns:
            str: The first valid word found, or an empty string if none found.
        """
        # Split the string on '\n'
        parts = input_string.split('\n')
        
        # Define the regex for a valid word
        valid_word_regex = r'^[A-Z][a-z]+.*$'
        
        # Iterate over parts to find the first valid word
        for part in parts:
            if re.match(valid_word_regex, part.strip()):
                return part.strip()
        
        return ""

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

        invalid_character_count = 0

        i = 0
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

            if cropped_pil is None:
                ocr_text = ""

            else:
                # cropped_pil.show()
                # i+= 1
                # if i > 6:
                #     quit()

                # Perform OCR on the cell
                ocr_text = pytesseract.image_to_string(cropped_pil, config="--psm 7").strip()

                # Deal with unwanted newline characters
                # ocr_text = self.split_and_get_valid_word(ocr_text)

                # Remove all characters until the first uppercase letter or accented uppercase letter
                ocr_text = re.sub(r'^[^A-ZÀ-Ö]*', '', ocr_text).strip()

                # Remove all characters at the end that are not letters or accented characters
                ocr_text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ]+$', '', ocr_text).strip()

                # If ocr_text is too short, then coerce to empty string
                if len(ocr_text) < 3:
                    print(f"Warning: Name string {ocr_text} is too short")
                    ocr_text = ""

                # If ocr_text is not a proper word, then return None
                if len(ocr_text) > 0 and not bool(re.match(r"^[A-Z]", ocr_text)):
                    print(f"Warning: Improper name... {ocr_text}")
                    return None
                
            # Save the OCR text to the dictionary
            cell_texts[(row, col)] = ocr_text

            invalid_character_count += self.count_non_letter_characters(ocr_text)

            if invalid_character_count > 15:
                return None
            
        return cell_texts


    def grid_number_from_image_text(self, text):
        """
        Extract the grid number from the OCR-extracted text.
        """
        match = re.search(r"GRID #(\d+)", text)
        return int(match.group(1)) if match else None


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
        if "date" in consolidated_metadata.columns:
            consolidated_metadata["date"] = consolidated_metadata["date"].astype(str)
        
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
        if "image_date" in parser_metadata.columns:
            parser_metadata["image_date"] = parser_metadata["image_date"].astype(str)

        # drop duplicates
        parser_metadata = parser_metadata.drop_duplicates(subset=["path"], keep="last")

        # Convert DataFrame to a list of dictionaries
        data_as_dicts = parser_metadata.to_dict(orient="records")

        # Write JSON with indentation
        with open(IMAGES_PARSER_PATH, "w") as f:
            json.dump(data_as_dicts, f, indent=4)

        print(f"Parser metadata saved: {IMAGES_PARSER_PATH}")


    def process_image_with_dynamic_grid(self, image_path, submitter, image_date, grid_number, matrix, skip_validation=False):
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
        print("Step 1: Finding logo...")
        logo_position = self.find_logo_position(image_path)

        print(f"Logo position: {logo_position}, image_path: {image_path}, grid_number: {grid_number}, submitter: {submitter}")

        if logo_position is None:
            parser_message = f"Warning: Failed to find logo position in {image_path}"
            return parser_message
        
        attempt = 0
        cell_scalars = [1, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.8]
        while attempt < len(cell_scalars):

            # Step 2: Get grid cells based on logo position
            print("Step 2: Dividing image into cells based on logo...")
            cell_scalar = cell_scalars[attempt]
            top_left = logo_position[0]
            bottom_right = logo_position[1]
            logo_mode = logo_position[2]
            grid_cells = self.divide_image_into_grid(image_path, top_left, bottom_right, cell_scalar, logo_mode)
            if not grid_cells:
                parser_message = f"Warning: Failed to divide grid cells in {image_path}"
                return parser_message

            # Step 3: Extract OCR results from each cell
            print("Step 3: Extracting text from image...")
            players_by_cell = self.extract_text_from_cells(image_path, grid_cells)

            # If unable to extract players_by_cell
            if players_by_cell is None:
                #self.draw_image_with_outlined_logo_and_grid_cells(image_path, logo_position, grid_cells)
                parser_message = f"Warning: Unable to extract player names from grid"
                print(parser_message)
                return parser_message
            
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
                cell_position = position_mapping.get((row, col))
                if cell_position:
                    responses[cell_position] = text

            # Step 4: Check that allocations match the text matrix
            matrix_flat = matrix_string_to_flat_list(matrix)
            responses_flat = list(responses.values())
            are_responses_robust = compare_flat_matrix_with_flat_image_responses(matrix_flat, responses_flat)

            if not are_responses_robust:
                # Issues with answers
                print(f"Issue with cell scalar {cell_scalar}!")
                print(matrix_flat)
                print(responses_flat)
                # self.draw_image_with_outlined_logo_and_grid_cells(image_path, logo_position, grid_cells, cell_scalar)
            else:
                # Valid answers were found, so exit loop
                break

            attempt += 1

        # self.draw_image_with_outlined_logo_and_grid_cells(image_path, logo_position, grid_cells, 1)
        # input("Press Enter to continue...")

        if not are_responses_robust and not skip_validation:
            print(f"Quitting on {grid_number} for {submitter}....")
            parser_message = f"Warning: Grid image is invalid"
            return parser_message
        elif not are_responses_robust and skip_validation:
            parser_message = "Warning: Grid validation skipped; proceeding with parsed responses."
        
        # Step 5: Save the processed image and metadata
        print("Step 5: Copying and saving image...")
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

        print("*" * 80)
        print("Success!!!")
        print(json.dumps(metadata, indent=4, ensure_ascii=False))
        print("*" * 80)

        self.save_consolidated_metadata(metadata)

        parser_message = f"Success: Processed image with grid number {grid_number}"
        print(f"Parsed metadata for {submitter} on {image_date}: grid {grid_number}")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

        return parser_message
    

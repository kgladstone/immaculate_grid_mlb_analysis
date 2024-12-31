import os
import sqlite3
import shutil
import pytesseract
import regex as re
import json
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np

from constants import GRID_PLAYERS, IMAGES_PATH, IMAGES_METADATA_PATH, LOGO_DARK_PATH, LOGO_LIGHT_PATH
from utils import ImmaculateGridUtils

# Global Tesseract config for OCR
OCR_CONFIG = r"--oem 3 --psm 6"

def crop_text_box_dynamic(image_path):
    """
    Detect and crop the text box in the given image dynamically.
    Args:
        image_path (str): Path to the input image.
    Returns:
        PIL.Image: Cropped text box as a PIL Image object, or None if no text box is detected.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return None

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


def find_logo_position(main_image_path, threshold=0.6):
    """
    Find the position of a target subimage (logo) within a main image.

    Args:
        main_image_path (str): Path to the main image.
        threshold (float): Matching confidence threshold (default: 0.6).

    Returns:
        tuple: (top_left, bottom_right) coordinates of the target subimage in the main image,
               or None if no match is found.
    """
    # Load the main image
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        print("Error: Unable to load the main image.")
        return None

    # Paths to logos
    logos = [LOGO_LIGHT_PATH, LOGO_DARK_PATH]

    for logo_path in logos:
        # Load the logo image
        logo_image = cv2.imread(logo_path)
        if logo_image is None:
            print(f"Error: Unable to load logo image at {logo_path}.")
            continue

        # Convert images to grayscale for template matching
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        logo_gray = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)

        # Match template
        result = cv2.matchTemplate(main_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Check if match is strong enough
        if max_val >= threshold:
            # Calculate bounding box of the detected logo
            logo_h, logo_w = logo_gray.shape
            top_left = max_loc
            bottom_right = (top_left[0] + logo_w, top_left[1] + logo_h)
            print(f"Logo found using {logo_path}.")
            return top_left, bottom_right

        print(f"Logo not found using {logo_path} (confidence: {max_val:.2f}).")

    # If no match is found
    print("Logo not found in either mode.")
    return None


def divide_image_into_grid(main_image_path, top_left, bottom_right):
    """
    Divide the main image into a 4x4 grid based on the size of the logo.

    Args:
        main_image_path (str): Path to the main image.
        top_left (tuple): (x, y) coordinates of the top-left corner of the logo.
        bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the logo.

    Returns:
        list: List of dictionaries containing top-left and bottom-right coordinates for each cell.
    """
    # Load the main image
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        print("Error: Unable to load the main image.")
        return []

    # Calculate the logo dimensions
    logo_width = bottom_right[0] - top_left[0]
    logo_height = bottom_right[1] - top_left[1]

    WIDTH_SCALAR = 1.7
    HEIGHT_SCALAR = 1.9
    TOP_ADJUSTMENT = 0.25
    LEFT_ADJUSTMENT = 0.25

    # Start grid at the top of the logo
    grid_start_x = top_left[0] - int(LEFT_ADJUSTMENT * logo_width)
    grid_start_y = top_left[1] - int(TOP_ADJUSTMENT * logo_height)

    grid_cells = []

    # Divide the image into a 4x4 grid based on the logo dimensions
    for row in range(4):
        for col in range(4):
            # Calculate the coordinates of the current cell
            x_start = grid_start_x + col * int(logo_width * WIDTH_SCALAR)
            x_end = x_start + int(logo_width * WIDTH_SCALAR)
            y_start = grid_start_y + row * int(logo_height * HEIGHT_SCALAR)
            y_end = y_start + int(logo_height * HEIGHT_SCALAR)

            # Ensure we don't exceed the image bounds
            if x_end > main_image.shape[1] or y_end > main_image.shape[0]:
                print(f"Skipping cell ({row}, {col}) - Out of bounds")
                continue

            # Append the cell's coordinates to the list
            if row > 0 and col > 0:
                grid_cells.append({
                    "row": row - 1,
                    "col": col - 1,
                    "top_left": (x_start, y_start),
                    "bottom_right": (x_end, y_end)
                })

    return grid_cells


def preprocess_image(img, crop_function):
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


def dynamic_crop_function(image_cv2):
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



def extract_text_from_image(image_path):
    """
    Extract text from an image using pytesseract.
    """
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img, config=OCR_CONFIG)
    except Exception as e:
        return f"Error extracting text: {e}"
    

def extract_text_from_cells(main_image_path, grid_cells):
    """
    Extract OCR text from each cell in the grid.

    Args:
        main_image_path (str): Path to the main image.
        grid_cells (list): List of dictionaries with cell coordinates (top-left, bottom-right).

    Returns:
        dict: A dictionary mapping each cell's (row, col) to its OCR text.
    """
    # Load the main image
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        print("Error: Unable to load the main image.")
        return {}

    cell_texts = {}

    # Process each cell
    for cell in grid_cells:
        row = cell["row"]
        col = cell["col"]
        top_left = cell["top_left"]
        bottom_right = cell["bottom_right"]

        # Crop the cell region from the image
        cropped_cell = main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Convert the cropped cell to a PIL image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cell, cv2.COLOR_BGR2RGB))
        cropped_pil = preprocess_image(cropped_pil, dynamic_crop_function)  # Apply preprocessing

        # Perform OCR on the cell
        ocr_text = pytesseract.image_to_string(cropped_pil, config=OCR_CONFIG).strip()

        # Save the OCR text to the dictionary
        cell_texts[(row, col)] = mlb_player_from_text(ocr_text)

    return cell_texts


def grid_number_from_image_text(text):
    """
    Extract the grid number from the OCR-extracted text.
    """
    match = re.search(r"BASEBALL GRID #(\d+)", text)
    return int(match.group(1)) if match else None


def mlb_player_from_text(text):
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



def save_consolidated_metadata(metadata, metadata_file):
    """
    Save or update the consolidated metadata JSON file, ensuring no duplicate entries
    for the same grid_number and submitter combination.
    """
    # Load existing metadata if the file exists
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            consolidated_metadata = json.load(f)
    else:
        consolidated_metadata = []

    # Check for existing data with the same grid_number and submitter
    existing_entry = next(
        (entry for entry in consolidated_metadata if
         entry["grid_number"] == metadata["grid_number"] and
         entry["submitter"] == metadata["submitter"]),
        None
    )

    if existing_entry:
        print(f"Metadata for grid_number {metadata['grid_number']} and submitter {metadata['submitter']} already exists.")
        return

    # Append the new metadata if no duplicate found
    consolidated_metadata.append(metadata)

    # Save the updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(consolidated_metadata, f, indent=4)

    print(f"Consolidated metadata saved: {metadata_file}")


def process_image_with_dynamic_grid(image_path, submitter, image_date, images_folder, metadata_file):
    """
    Process an image with dynamic header/footer detection and grid cell OCR assignment.

    Args:
        image_path (str): Path to the input image.
        submitter (str): Name of the submitter.
        image_date (str): Date of the image.
        images_folder (str): Folder to save the processed images.
        metadata_file (str): Path to the consolidated metadata file.

    Returns:
        int: Grid number if successfully processed, None otherwise.
    """

    # Step 1: Find the logo position
    position = find_logo_position(image_path)
    if position is None:
        print(f"Failed to find logo position in {image_path}")
        return None

    # Step 2: Get grid cells based on logo position
    grid_cells = divide_image_into_grid(image_path, position[0], position[1])
    if not grid_cells:
        print(f"Failed to divide grid cells in {image_path}")
        return None

    # Step 3: Extract OCR results from each cell
    players_by_cell = extract_text_from_cells(image_path, grid_cells)

    # # Show any subimages that did not extract text
    # for cell, text in players_by_cell.items():
    #     if text is None:
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
    text = extract_text_from_image(image_path)
    grid_number = grid_number_from_image_text(text)
    if grid_number is None:
        print(f"Failed to extract grid number from {image_path}")
        return None

    # Step 5: Save the processed image and metadata
    dest_filename = f"{submitter}_{image_date}_grid_{grid_number}.jpg"
    dest_image_path = os.path.join(images_folder, dest_filename)
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

    save_consolidated_metadata(metadata, metadata_file)

    return grid_number


def refresh_image_data(images_save_folder, metadata_file, GRID_PLAYERS):
    """
    Refresh the image folder by copying screenshots from Messages to a specified folder.
    """
    # Paths to Messages database and attachments folder
    db_path = os.path.expanduser("~/Library/Messages/chat.db")
    attachments_base_path = os.path.expanduser("~/Library/Messages/Attachments/")
    
    # Map phone numbers to player names
    phone_to_player = {details["phone_number"]: player for player, details in GRID_PLAYERS.items()}
    phone_numbers = list(phone_to_player.keys())
    
    # Ensure the save folder exists
    os.makedirs(images_save_folder, exist_ok=True)

    # Connect to the Messages database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Query to find attachments with metadata
        query = f"""
        SELECT attachment.filename, attachment.mime_type, message.date, handle.id
        FROM message
        JOIN handle ON message.handle_id = handle.ROWID
        JOIN message_attachment_join ON message.ROWID = message_attachment_join.message_id
        JOIN attachment ON message_attachment_join.attachment_id = attachment.ROWID
        WHERE handle.id IN ({','.join(['?' for _ in phone_numbers])})
          AND (attachment.mime_type LIKE 'image/png'
               OR attachment.mime_type LIKE 'image/jpeg')
          AND (attachment.filename LIKE '%IMG_%' 
               OR attachment.filename LIKE '%Screenshot%')
        """
        cursor.execute(query, phone_numbers)
        results = cursor.fetchall()

        # Process each attachment
        for filename, mime_type, message_date, sender_phone in results:
            if filename:
                print("*" * 40)

                # Resolve the source path
                if filename.startswith("~/Library/Messages/Attachments/"):
                    filename = filename.replace("~/Library/Messages/Attachments/", "")
                src_path = os.path.join(attachments_base_path, filename.lstrip("/"))

                if os.path.exists(src_path):
                    submitter = phone_to_player.get(sender_phone, "Unknown")
                    image_date = ImmaculateGridUtils._convert_timestamp(message_date)
                    grid_number = process_image_with_dynamic_grid(src_path, submitter, image_date, images_save_folder, metadata_file)

                    if grid_number is None:
                        print(f"Could not extract grid number from {src_path}")

        print(f"Screenshots saved to {images_save_folder}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        conn.close()


# Usage example
refresh_image_data(IMAGES_PATH, IMAGES_METADATA_PATH, GRID_PLAYERS)

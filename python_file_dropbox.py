# Import standard libraries
import os
import io
import numpy as np
import pandas as pd
import cv2
from time import sleep
import re
from typing import Optional, Any, List, Tuple, Union, Dict
from time import time
import logging
import shutil
from glob import glob
from os.path import join


# Import Google Cloud libraries
from google.cloud import vision
from google.cloud.vision_v1 import types

# Import GUI automation libraries
import pyautogui
import pyperclip

# Set environment variable for Google Cloud
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = r"C:\Users\david\Google Drive\ocr-project-281722-2224f2068b64.json"


logging.basicConfig(
    filename=r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\log.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True  # Ensure that the handler is applied, even if one exists
)


# Configuration
SCREENSHOT_PATH = r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\GTO_Wizard_Macro_1.png"
RELATIVE_POSITION_PATH = r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\node.png"
ACTION_PATH = r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\action.png"
IP_CROWN_PATH = r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\active_player_crown.png"
NODE_IMAGE_DST = r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\node.png"
GOOD_GAME_PATH = r"C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\GG.png"
START_Y = 100
END_Y = 438
WINDOW_SIZE = 150
MIN_BLUE = 0
MIN_GREEN = 200
MIN_RED = 0
MAX_BLUE = 255
MAX_GREEN = 255
MAX_RED = 255


def find_subimage(
    sub_image_path: str, screenshot_filepath: str, time_seconds: int
) -> Tuple[float, float]:
    """
    Finds the subimage in the screenshot and returns the coordinates of the center
    of the subimage
    """
    sub_image = cv2.imread(sub_image_path, 0)
    x, y = "Not Found", "Not Found"
    for _ in range(10 * time_seconds):
        sleep(0.1)
        full_image = cv2.imread(screenshot_filepath, 0)
        res = cv2.matchTemplate(full_image, sub_image, cv2.TM_CCOEFF_NORMED)
        confidence = cv2.minMaxLoc(res)[1] > 0.8
        if not confidence:
            continue
        img2 = full_image.copy()
        w, h = sub_image.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = [
            "cv2.TM_CCOEFF",
            "cv2.TM_CCOEFF_NORMED",
            "cv2.TM_CCORR",
            "cv2.TM_CCORR_NORMED",
            "cv2.TM_SQDIFF",
            "cv2.TM_SQDIFF_NORMED",
        ]

        for meth in methods:
            img = img2.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img, sub_image, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            top_left = (
                min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
            )
            bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        top_left_x = [top_left[0]]
        top_left_y = [top_left[1]]
        bottom_right_x = [bottom_right[0]]
        bottom_right_y = [bottom_right[1]]
        top_left_x = np.median(top_left_x)
        top_left_y = np.median(top_left_y)
        bottom_right_x = np.median(bottom_right_x)
        bottom_right_y = np.median(bottom_right_y)
        x = (top_left_x + bottom_right_x) / 2
        y = (top_left_y + bottom_right_y) / 2
        break
    return (x, y)


class OCR:
    """A class for performing optical character recognition (OCR) on images."""

    def __init__(self):
        """Creates a TextDetector object.

        Args:
            None.

        Returns:
            A TextDetector object.
        """
        # Create a Cloud Vision ImageAnnotatorClient object
        self.client = vision.ImageAnnotatorClient()

    def read_image(self, path: str) -> types.Image:
        """Loads an image from a file.

        Args:
            path: The path to the image file.

        Returns:
            A Cloud Vision Image object.
        """

        # Open the image file and read its contents
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        # Create a Cloud Vision Image object from the image contents
        return types.Image(content=content)

    def get_text_annotations(self, image: types.Image) -> List[vision.TextAnnotation]:
        """Detects text in an image.

        Args:
            image: A Cloud Vision Image object.

        Returns:
            A list of text annotations.
        """

        # Detect text in the image
        response = self.client.text_detection(image=image)
        # Return the list of text annotations
        return response.text_annotations

    def get_ocr_and_coords(
        self, texts: List[vision.TextAnnotation]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Extracts the text and coordinates for each detected text region.

        Args:
            texts: A list of text annotations.

        Returns:
            A tuple of two lists: the first list contains the text for each detected
            text region, and the second list contains the coordinates for each detected
            text region.
        """

        # Extract the text and coordinates for each detected text region
        ocr = [text.description for text in texts]
        coords = [
            [[vertex.x, vertex.y] for vertex in text.bounding_poly.vertices]
            for text in texts
        ]
        coords = [np.array(x) for x in coords]
        # Return the two lists
        return ocr, coords

    def get_boundaries(
        self, coords: List[np.ndarray]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Gets the bounding box for each set of coordinates.

        Args:
            coords: A list of NumPy arrays containing the coordinates for each detected
            text region.

        Returns:
            A tuple of four lists: the first list contains the x-coordinates of the
            top-left corner of each bounding box, the second list contains the
            y-coordinates of the top-left corner of each bounding box, the third list
            contains the x-coordinates of the bottom-right corner of each bounding box,
            and the fourth list contains the y-coordinates of the bottom-right corner
            of each bounding box.
        """

        # Get the x-coordinates of the top-left corner of each bounding box
        x0 = [int(x[:, 0].min()) for x in coords]
        # Get the y-coordinates of the top-left corner of each bounding box
        x1 = [int(x[:, 0].max()) for x in coords]
        # Get the x-coordinates of the bottom-right corner of each bounding box
        y0 = [int(x[:, 1].min()) for x in coords]
        # Get the y-coordinates of the bottom-right corner of each bounding box
        y1 = [int(x[:, 1].max()) for x in coords]
        # Return the four lists
        return x0, x1, y0, y1

    def detect_text(self, path: str) -> np.ndarray:
        """Detects text in an image.

        Args:
            path: The path to the image file.

        Returns:
            A NumPy array containing the detected text, as well as the bounding boxes
            for each text region.
        """

        # Load the image from the file
        image = self.read_image(path)
        # Detect text in the image
        texts = self.get_text_annotations(image)
        # Extract the text and coordinates for each detected text region
        ocr, coords = self.get_ocr_and_coords(texts)
        # Get the bounding boxes for each set of coordinates
        boundaries = self.get_boundaries(coords)
        # Create a NumPy array containing the detected text, as well as the bounding
        # boxes for each text region
        return np.array(list(zip(ocr, *boundaries)))


google_ocr = OCR()


def is_numeric(s: str) -> bool:
    """
    Returns True if the input string is numeric.

    Args:
        s: The input string.

    Returns:
        True if the input string is numeric, False otherwise.
    """

    try:
        float(s)
        return True
    except ValueError:
        return False


def swap_and_join(input_string: str) -> str:
    """
    Swaps the first two items in the input string and returns the joined string.

    Args:
        input_string: The input string.

    Returns:
        The joined string with the first two items swapped.
    """

    # Split the input string into a list of words
    split_string = input_string.split(" ")

    # Check if the input string has two words
    if len(split_string) == 2:
        # Get the first two words
        first_item, second_item = split_string

        # Check if the first word is a number
        if is_numeric(first_item):
            # Swap the first two words
            split_string[0], split_string[1] = second_item, first_item
    # Join the list of words back into a string
    return " ".join(split_string)


class OCRMerger:
    """
    This class merges OCR values of an input array.
    """

    def __init__(self, data: np.ndarray):
        """
        Initializes the `OCRMerger` class.

        Args:
            data: The input array.
        """
        self.data = data

    def sort_data_by_y_start(self) -> np.ndarray:
        """
        Sorts the input array by y_start values.

        Returns:
            The sorted input array.
        """
        return self.data[self.data[:, 3].astype(int).argsort()]

    def merge_group(self, group: List[np.ndarray]) -> List:
        """
        Merges the OCR values of the input group.

        Args:
            group: The input group.

        Returns:
            The merged OCR values.
        """
        merged_ocr_value = " ".join(r[0] for r in group)
        return [merged_ocr_value] + list(group[-1][1:])

    def merge_ocr_values(self) -> np.ndarray:
        """
        Merges the OCR values of the input array.

        Returns:
            The merged OCR values.
        """
        # Sort the input array by y_start values
        sorted_data = self.sort_data_by_y_start()
        # Initialize a list to store the merged OCR values
        merged_data = []
        # Initialize a list to store the current group of OCR values
        current_group = []

        # Iterate over the sorted input array
        for row in sorted_data:
            # Get the OCR value, x_start, x_end, y_start, and y_end from the row
            ocr_value, x_start, x_end, y_start, y_end = row
            # Convert the y_start and y_end values to integers
            y_start, y_end = int(y_start), int(y_end)

            # If the current group is empty, add the row to the current group
            if not current_group:
                current_group.append(row)
            else:
                # Get the last row in the current group
                last_row = current_group[-1]
                # Get the y_start and y_end values from the last row
                last_y_start, last_y_end = int(last_row[3]), int(last_row[4])

                # If the y_start of the current row is less than or equal to the y_end
                # of the last row, add the row to the current group.
                if y_start <= last_y_end:
                    current_group.append(row)
                else:
                    # Merge the OCR values in the current group and add the merged OCR
                    # values to the list of merged OCR values
                    merged_data.append(self.merge_group(current_group))
                    # Clear the current group
                    current_group = [row]

        # If the current group is not empty, merge the OCR values in the current group
        # and add the merged OCR values to the list of merged OCR values.
        if current_group:
            merged_data.append(self.merge_group(current_group))

        # Return the list of merged OCR values
        return np.array(merged_data)



def show_image(image: np.ndarray, title: str = "Image") -> None:
    """
    Shows the image using OpenCV
    """
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


class ActivePlayerROI:
    """
    This class provides methods for extracting the active player's region of interest
    (ROI) from a screenshot of a game.
    """
    def __init__(self, node_name: str):
        """
        Initializes the `ActivePlayerROI` class.
        """
        self.node_name = node_name

    @staticmethod
    def convolution_comparison(
        arr: np.ndarray, kernel: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Performs convolution and comparison with the given kernel and array.

        Args:
            arr: The input array.
            kernel: The kernel array.
            window_size: The size of the window.

        Returns:
            The output array.
        """
        # Convolve the input array with the kernel
        conv = np.convolve(arr, kernel, mode="valid")
        # Find the locations where the convolution matches the window_size
        loc = np.where(conv == window_size)[0]
        # Return the locations
        return loc

    def find_first_instance(
        self, arr: np.ndarray, window_size: int, horizontal: bool
    ) -> Tuple[int, int]:
        """
        Finds the first instance of consecutive 1's of given window_size either
        horizontally or vertically.

        Args:
            arr: The input array.
            window_size: The size of the window.
            horizontal: Whether to search horizontally or vertically.

        Returns:
            The coordinates of the first instance of consecutive 1's.
        """
        # Create a kernel of all 1's of size window_size
        kernel = np.ones(window_size)
        # Initialize the variable to store the coordinates of the first instance
        first_instance = None

        # If the `horizontal` flag is True, search the input array horizontally
        if horizontal:
            for row_idx, row in enumerate(arr):
                # Convolve the current row with the kernel
                # Check if the convolution matches the window_size
                loc = self.convolution_comparison(row, kernel, window_size)

                # If a match was found, store the coordinates
                if loc.size > 0:
                    first_instance = (row_idx, loc[0])
                    break
        else:
            for col_idx in range(arr.shape[1]):
                # Convolve the current row with the kernel
                # Check if the convolution matches the window_size
                column = arr[:, col_idx]
                loc = self.convolution_comparison(column, kernel, window_size)

                # If a match was found, store the coordinates
                if loc.size > 0:
                    first_instance = (col_idx, loc[0])
                    break
        return first_instance

    def find_first_instance_both(
        self, arr: np.ndarray, window_size: int = 150
    ) -> Tuple[int, int]:
        """
        Finds the first instance of 150 consecutive 1's horizontally and vertically.

        Args:
            arr: The input array.
            window_size: The size of the window.

        Returns:
            The coordinates of the first instance of 150 consecutive 1's.
        """
        # find the first instance of 150 consecutive 1's horizontally
        horizontal_first_instance = self.find_first_instance(
            arr, window_size, horizontal=True
        )

        # find the first instance of 150 consecutive 1's vertically
        vertical_first_instance = self.find_first_instance(
            arr, window_size, horizontal=False
        )

        return (horizontal_first_instance[1], vertical_first_instance[1])


    def extract_active_player_roi(self, save_screenshots=False, test_mode=False, test_files=None) -> Tuple[Tuple[int, int], np.ndarray, Dict[str, Any]]:
        """
        Extracts the active player's ROI from a screenshot of a game.

        Args:
            save_screenshots: If True, save the screenshots.
            test_mode: If True, use saved screenshots instead of taking new ones.
            test_files: Dictionary containing the test files.

        Returns:
            A tuple of the coordinates of the ROI, the screenshot, and the updated test_files.
        """
        if test_mode:
            img_screenshot = cv2.imread(test_files['extract_active_player_roi'].pop(0))
        else:
            # save the screenshot
            _ = pyautogui.screenshot(SCREENSHOT_PATH)
            img_screenshot = cv2.imread(SCREENSHOT_PATH)
            if save_screenshots:
                save_file_func(node_name=self.node_name, data=img_screenshot, info="ActivePlayerRoi", filetype="png")



        # crop the screenshot to the active player's region
        img = img_screenshot[START_Y:END_Y, :]
        # Convert the image to HSV
        image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Convert the image to black and white using the given mask
        mask = cv2.inRange(
            image_HSV, (MIN_BLUE, MIN_GREEN, MIN_RED), (MAX_BLUE, MAX_GREEN, MAX_RED)
        )
        # Convert the image to 1's and 0's
        data = (mask == 255).astype(int)

        # Find the first instance of 100 consecutive 1's horizontally and vertically
        active_roi = self.find_first_instance_both(data, window_size=WINDOW_SIZE)
        # Adjust the coordinates to account for the cropping
        active_roi = (active_roi[0], active_roi[1] + START_Y)
        # return the coordinates and the screenshot
        return active_roi, img_screenshot, test_files



class ActionOptionsExtractor:
    def __init__(self, img_screenshot: np.ndarray, active_roi: Tuple[int, int]):
        self.img_screenshot = img_screenshot
        self.active_roi = active_roi

    def crop_and_save_node_image(self) -> None:
        """
        This function crops and saves the node image from the screenshot.

        Args:
            img_screenshot: The screenshot of the game.
            active_roi: The coordinates of the active player's ROI.
        """

        # Get the start and end coordinates of the node image.
        start_x, start_y = self.active_roi
        end_y, end_x = start_y + 180, start_x + 150

        # Crop the screenshot to the node image.
        node_image = self.img_screenshot[start_y:end_y, start_x:end_x]

        # Save the node image to a file.
        cv2.imwrite(NODE_IMAGE_DST, node_image)

    def extract_actions(self, ocr: np.ndarray) -> List[str]:
        """
        This function extracts the actions from the OCR results.

        Args:
            ocr: The OCR results.

        Returns:
            A list of actions.
        """
        # Get the list of options the player has
        return [
            action
            for action in ["FOLD", "CALL", "RAISE", "ALLIN"]
            if action in [x[0] for x in ocr[1:]]
        ]

    def adjust_ocr_coordinates(self, ocr: np.ndarray) -> np.ndarray:
        """
        This function adjusts the coordinates of the OCR results to be relative to
        the active player's ROI.

        Args:
            ocr: The OCR results.
            active_roi: The coordinates of the active player's ROI.

        Returns:
            The adjusted OCR results.
        """
        # Create a new array with the adjusted coordinates
        new_arr = [
            [
                arr[0],
                self.active_roi[0] + int((int(arr[1]) + int(arr[2])) / 2),
                self.active_roi[1] + int((int(arr[3]) + int(arr[4])) / 2),
            ]
            for arr in ocr
        ]
        return new_arr

    def extract_position(self, ocr: np.ndarray) -> str:
        """
        This function extracts the position from the OCR results.

        Args:
            ocr: The OCR results.

        Returns:
            The position.
        """
        # Get the position from the first OCR result
        position = ocr[0][0].split(" ")[0]
        # If the position is "UTG", replace it with "LJ"
        # If the position is "BTN", replace it with "BN"
        return "LJ" if position == "UTG" else "BN" if position == "BTN" else position

    def extract_action_options(self) -> Dict:
        """
        This function extracts the options available to the player from the screenshot.

        Args:
            active_roi: The coordinates of the active player's ROI.
            img_screenshot: The screenshot of the game.

        Returns:
            A dictionary of the options available to the player.
        """
        # Crop and save the node image
        self.crop_and_save_node_image()

        # Get the OCR results for the node image
        ocr = google_ocr.detect_text(NODE_IMAGE_DST)
        # Merge the OCR results
        ocr = OCRMerger(ocr[1:]).merge_ocr_values()
        # Extract the actions from the OCR results
        actions = self.extract_actions(ocr)
        # Get the number of actions
        number_of_options = len(actions)
        # Adjust the coordinates of the OCR results
        ocr = self.adjust_ocr_coordinates(ocr)

        # Swap the words "and" and "to" in the OCR results
        for num, arr in enumerate(ocr):
            ocr[num][0] = swap_and_join(arr[0])
        # Extract the position from the OCR results
        position = self.extract_position(ocr)

        # Create a dictionary of the options available to the player
        ocr_dict = {
            "position": position,
            "number_of_options": number_of_options,
            "actions": actions,
            "ocr": ocr,
        }

        # Log the dictionary
        logging.info(f"Info Gathering Dictionary: {ocr_dict}\n\n")

        # Return the dictionary
        return ocr_dict


class ActionExtractor:
    def __init__(self):
        pass

    @staticmethod
    def contains_alphabets(input_str: str) -> bool:
        """
        Check if the input string contains any alphabetic characters.

        Args:
            input_str: The input string to check.

        Returns:
            True if the input string contains any alphabetic characters, False otherwise.
        """
        return any(char.isalpha() for char in input_str)

    @staticmethod
    def contains_digits(input_str: str) -> bool:
        """
        Check if the input string contains any numeric characters.

        Args:
            input_str: The input string to check.

        Returns:
            True if the input string contains any numeric characters, False otherwise.
        """
        return any(char.isdigit() for char in input_str)

    @staticmethod
    def contains_action(input_str: str) -> bool:
        """
        Check if the input string contains any action keywords.

        Args:
            input_str: The input string to check.

        Returns:
            True if the input string contains any action keywords, False otherwise.
        """
        keywords = ("FOLD", "CALL", "RAISE", "ALLIN")
        return any(keyword in input_str.upper() for keyword in keywords)

    @staticmethod
    def classify_text(input_str: str) -> bool:
        """
        Classify the input string to check if it contains an action or a numeric value.

        Args:
            input_str: The input string to check.

        Returns:
            True if the input string contains an action or a numeric value, False otherwise.
        """
        if contains_action_bool := ActionExtractor.contains_action(input_str):
            return True
        contains_text_bool = ActionExtractor.contains_alphabets(input_str)
        contains_numbers_bool = ActionExtractor.contains_digits(input_str)
        return bool(contains_numbers_bool and not contains_text_bool)

    @staticmethod
    def crop_copy_nodes(range_side: str) -> List[List[Union[str, int]]]:
        """
        Crops the text for copying ranges from the screenshot

        Args:
            range_side: The side of the range to crop, either "left" or "right".

        Returns:
            A list of cropped OCR values with adjusted coordinates.
        """
        _ = pyautogui.screenshot(SCREENSHOT_PATH)
        if range_side == "left":
            offset = 0
            action_img = cv2.imread(SCREENSHOT_PATH)[420:700, offset:100]
        else:
            offset = 2135
            action_img = cv2.imread(SCREENSHOT_PATH)[420:700, offset : (offset + 100)]

        _ = cv2.imwrite(ACTION_PATH, action_img)

        action_ocr = google_ocr.detect_text(ACTION_PATH)
        action_ocr = np.array(
            [arr for arr in action_ocr[1:] if ActionExtractor.classify_text(arr[0])],
            dtype="<U100",
        )
        action_ocr = OCRMerger(action_ocr).merge_ocr_values()
        action_ocr = [
            arr
            for arr in action_ocr
            if any(
                word in arr[0].upper() for word in ["FOLD", "CALL", "RAISE", "ALLIN"]
            )
        ]
        new_arr = []
        for arr in action_ocr:
            l = [
                arr[0],
                offset + int((int(arr[1]) + int(arr[2])) / 2),
                420 + int((int(arr[3]) + int(arr[4])) / 2),
            ]
            new_arr.append(l)
        return new_arr

action_extractor = ActionExtractor()

class RangeCopier:
    def __init__(self):
        pass

    @staticmethod
    def determine_active_oop_or_ip(img_screenshot: np.ndarray) -> str:
        """
        Determines if the active player's copy button are on the left or right side
        of the screen
        """
        _ = cv2.imwrite(RELATIVE_POSITION_PATH, img_screenshot)
        output = find_subimage(IP_CROWN_PATH, RELATIVE_POSITION_PATH, 3)
        if output == ("Not Found", "Not Found"):
            raise Exception("Could not find opponent text")
        elif output[0] < 1000:
            return "left"
        else:
            return "right"


    @staticmethod
    def move_to_side(action_ocr: List, range_side: str) -> None:
        """
        Moves the mouse cursor to the side of the specified action OCR.

        Args:
            action_ocr: A list of two integers, where the first integer is the
            x-coordinate and the second integer is the y-coordinate of the action OCR.
            range_side: A string that specifies the side of the action OCR to move the
            mouse cursor to. Valid values are `left` and `right`.

        Returns:
            None
        """
        # Move the mouse cursor to the last OCR result
        pyautogui.moveTo(action_ocr[-1][1], action_ocr[-1][2])

        # If the range side is 'left', move the mouse cursor to the left edge of the
        # OCR result
        if range_side == "left":
            x, y = pyautogui.position()
            pyautogui.moveTo(1, y)
            pyautogui.moveTo(1, action_ocr[-1][2] + 3)
        # If the range side is 'right', move the mouse cursor to the right edge of the
        # OCR result
        else:
            pyautogui.moveTo(min(action_ocr[-1][1] + 50, 2226), action_ocr[-1][2])

    @staticmethod
    def copy_action_ranges(
        action_ocr: List, copy_btn_loc: Tuple[int, int]
    ) -> Dict[str, str]:
        """
        Copies the action ranges from the screen to the clipboard.

        Args:
            action_ocr: A list of OCR results, where each result is a list of three
                strings:[action, start, end].
            copy_btn_loc: The location of the copy button on the screen, as a tuple of
                two integers: (x, y).

        Returns:
            A dictionary mapping action names to their corresponding ranges.
        """
        # Create a dictionary to store the action ranges
        range_dict = {}

        # Iterate over the OCR results
        for arr in action_ocr:
            # Get the x and y coordinates of the action range
            x, y = arr[1], arr[2]
            # Move the mouse cursor to the action range
            pyautogui.moveTo(x, y)
            # Click the action range
            pyautogui.click()
            # Move the mouse cursor to the copy button
            pyautogui.moveTo(copy_btn_loc)

            # TODO: Make this dynamic
            sleep(2.5)
            # Get the text that is currently selected on the screen
            range_dict[arr[0]] = pyperclip.paste()
            # Sleep for 0.5 seconds
            sleep(0.5)
        return range_dict


    def copy_ranges(cls, img_screenshot: np.ndarray, ocr_dict: Dict[str, Any], current_node: str) -> Tuple[str, int, Dict[str, str], List[str]]:
        """
        Copies the action ranges from the screen to the clipboard, and returns the
        current node, position, and a dictionary mapping action names to their
        corresponding ranges.

        Args:
            img_screenshot: The screenshot of the screen, as an array of numbers.
            ocr_dict: A dictionary containing the OCR results for the screenshot,
            where the keys are the x and y coordinates of the OCR results, and the
            values are the text that was recognized at those coordinates.

            current_node: The name of the current node.

        Returns:
            A tuple of four values: the current node, the position of the current node,
            a dictionary mapping action names to their corresponding ranges, and a list
            of the OCR results that were not action ranges.
        """
        # Get the position of the current node
        position = ocr_dict["position"]
        # Get the OCR results for the screenshot
        ocr = ocr_dict["ocr"]

        # Determine which side of the screen is currently active
        range_side = cls.determine_active_oop_or_ip(img_screenshot)
        # Get the location of the copy button
        side_dict = {"left": (78, 445), "right": (2221, 440)}
        copy_btn_loc = side_dict[range_side]

        # Move the mouse cursor to the copy button
        pyautogui.moveTo(copy_btn_loc)
        # Sleep for 1 second
        sleep(1)

        # Get the OCR results for the action ranges
        action_ocr = action_extractor.crop_copy_nodes(range_side)
        # Move the mouse cursor to the side of the screen where the action ranges are
        # located
        cls.move_to_side(action_ocr, range_side)
        # Sleep for 1 second
        sleep(1)
        # Get the OCR results for the action ranges again
        action_ocr = action_extractor.crop_copy_nodes(range_side)

        # For each action range, swap the first two words and join them together
        for i in range(len(action_ocr)):
            action_ocr[i][0] = swap_and_join(action_ocr[i][0])
        # Reverse the order of the action ranges
        _ = action_ocr.reverse()
        # Copy the action ranges to the clipboard
        range_dict = cls.copy_action_ranges(action_ocr, copy_btn_loc)

        # Return the current node, position, range_dict, and a list of the OCR results
        # that were not action ranges
        return (current_node, position, range_dict, ocr[1:])


range_copier = RangeCopier()





class PokerNodeAnalyzer:
    def __init__(self):
        self.screenshot_path = SCREENSHOT_PATH
        self.ip_crown_path = IP_CROWN_PATH

    def generate_node(self, node: str, position: str, action: str) -> str:
        """
        Generate a node given a node, position, and action.

        Args:
        node (str): The node to generate.
        position (str): The position of the node.
        action (str): The action to perform on the node.

        Returns:
        The generated node.

        Raises:
        ValueError: If the node is not a valid node.
        ValueError: If the position is not a valid position.
        ValueError: If the action is not a valid action.
        """
        # Reset node if it is the first action
        node = self.clear_first_action_node(node)

        # Generate action string based on the action input
        action_str = self.generate_action_str(node, action)

        # Create the output node
        output_node = self.create_output_node(node, position, action_str)

        # Update raise abbreviations in the output node
        output_node = self.update_raise_abbreviations(output_node)

        return output_node

    def clear_first_action_node(self, node: str) -> str:
        """
        Clear the node if it is the first action
        """
        return "" if node == "FirstAction" else node

    def generate_action_str(self, node: str, action: str) -> str:
        """
        Generate a string representation of an action.

        Args:
        node (str): The node to generate the action for.
        action (str): The action to generate.

        Returns:
        The string representation of the action.

        Raises:
        Exception: If the action is not recognized.

        Comments:

        * The `if` statement checks if the action is `FOLD`. If it is, the function
            returns the string `f`.
        * The `elif` statement checks if the action contains the string `CALL`. If
            it does, the function calls the `find_largest_number()` function to find
            the largest number in the node. If the number is not found, the function
            returns the string `1.00`. The function then returns the string `c` followed
            by the amount.
        * The `elif` statement checks if the action contains the string `RAISE` or
            `ALLIN`.
            If it does, the function calls the `generate_raise_action_str()` function to
            generate the string representation of the raise.
        * The `else` statement raises an exception if the action is not recognized.
        """
        if action.upper() == "FOLD":
            return "f"
        elif "CALL" in action.upper():
            amount = self.find_largest_number(node) or 1.00
            return f"c{amount}"
        elif "RAISE" in action.upper() or "ALLIN" in action.upper():
            return self.generate_raise_action_str(node, action)
        else:
            raise Exception("Action not recognized")

    def generate_raise_action_str(self, node: str, action: str) -> str:
        """
        Generate the raise action string based on the action input
        """
        amount = action.split(" ")[-1]
        max_bet_infront = str(self.find_largest_number(node))
        return f"c{amount}" if max_bet_infront == amount else f"b{amount}"

    def create_output_node(self, node: str, position: str, action_str: str) -> str:
        """
        Create the output node based on the input node, position, and action string
        """
        if not node:
            return f"{position}_{action_str}"
        output_node = "__|__".join([node, f"{position}_{action_str}"])
        return output_node.replace("o", "O")

    def not_end_node(self) -> bool:
        """
        Checks if the screenshot contains the end node
        """
        _ = pyautogui.screenshot(SCREENSHOT_PATH)
        output = find_subimage(IP_CROWN_PATH, SCREENSHOT_PATH, 1)
        return output != ("Not Found", "Not Found")

    def find_xth_occurrence(self, text: str, char: str, x: int) -> int:
        """
        Finds the index of the Xth occurrence of a character in a string
        """
        count = 0
        for index, letter in enumerate(text):
            if letter == char:
                count += 1
                if count == x:
                    return index
        return -1  # If the Xth occurrence is not found

    def reduce_number(self, input_str: str) -> str:
        def _replace(match):
            return f"_ {int(match.group(1)) % 10}b"

        pattern = r"_(\d+)b"
        result = re.sub(pattern, _replace, input_str)
        return result.replace(" ", "")

    def update_raise_abbreviations(self, node_text: str) -> str:
        """
        Updates the raise abbreviations in the node text - 3b ... 6b etc
        """
        occurences = node_text.count("b")
        if occurences not in [0, 1]:
            for i in range(occurences, occurences - 1, -1):
                index = self.find_xth_occurrence(node_text, "b", i)
                node_text = node_text[:index] + f"{i+1}b" + node_text[index + 1 :]
        return self.reduce_number(node_text)

    def find_largest_number(self, text: str) -> Union[float, None]:
        """
        Finds the largest number in a string
        """
        pattern = r"[b|c]([\d.]+)"
        matches = re.findall(pattern, text)
        numbers = [float(num) for num in matches]
        return max(numbers, default=None)

    def new_nodes(
        self, node: str, position: int, node_branches: List[List[Any]]
    ) -> List[List[Any]]:
        """
        Generates the new nodes for the list of player options
        """
        output_l: List[List[Any]] = []
        for l in node_branches:
            action = l[0]
            x_mid = l[1]
            y_mid = l[2]
            new_node = self.generate_node(node, position, action)
            output_l.append([new_node, x_mid, y_mid])
        return output_l

    def replace_cyrillic_ve_with_b(self, text: str) -> str:
        """
        Replace Cyrillic letter 've' with 'b'.

        Args:
        text (str): The text to replace the Cyrillic letter in.

        Returns:
        The text with the Cyrillic letter replaced.

        Comments:

        * The `replace()` method replaces all instances of the Cyrillic letter 've'
            (character code 1042) with the letter 'b'.
        """
        return text.replace(chr(1042), "B")

    def is_end_point_reached(self, node: str) -> bool:
        """
        Determine if the node is an end point
        """
        print("analyzing node string: ", node)
        pending_to_act_l = ["LJ", "HJ", "CO", "BN", "SB", "BB"]
        acted_and_headed_to_flop_l = []
        out_of_hand_l = []

        actions_list = node.split("__|__")
        for action_str in actions_list:
            position, action = self.parse_position_and_action(action_str)

            pending_to_act_l.remove(position)

            if "b" in action:
                bet_level = self.get_bet_level(action)
                if bet_level >= 3:
                    pending_to_act_l.extend(iter(acted_and_headed_to_flop_l))
                acted_and_headed_to_flop_l = [position]
            if action.startswith("f"):
                out_of_hand_l.append(position)
            if action.startswith("c"):
                acted_and_headed_to_flop_l.append(position)
        return self.check_if_end_point(pending_to_act_l, acted_and_headed_to_flop_l)

    def parse_position_and_action(self, action_str: str) -> Tuple[str, str]:
        """
        Parse the position and action from the action string
        """
        position, action = action_str.split("_")
        position = self.replace_cyrillic_ve_with_b(position)
        return position, action

    def get_bet_level(self, action: str) -> int:
        """
        Determine the bet level of the action
        """
        return 2 if action[0] == "b" else int(action[0])

    def check_if_end_point(
        self, pending_to_act_l: List[str], acted_and_headed_to_flop_l: List[str]
    ) -> bool:
        """
        Check if the current node is an end point
        """
        if not pending_to_act_l:
            return True
        elif not acted_and_headed_to_flop_l and len(pending_to_act_l) == 1:
            return True
        else:
            return False

    def click_node_start(self):
        pyautogui.click(292, 342)
        sleep(2)

    def ignore_node_check(self, node):
        # determine how many players have vpip'd
        vpip_count = len(
            {
                pos.split("_")[0]
                for pos in [x for x in node.split("__|__") if "_f" not in x]
            }
        )

        # determine if the node contains a 5-bet
        five_bet_present = "5b" in node

        ignore_node = vpip_count > 3 or five_bet_present
        logging.info(f"ignore_node: {ignore_node}")
        return ignore_node


poker_node_analyzer = PokerNodeAnalyzer()


SAVE_SCREENSHOTS_FLDR = r'C:\Users\david\OneDrive\Desktop\D_Drive_BckUp\ICM_Calc\HyperTurboResearch\VideoFrames\Bovada\GUI\GTO_Wizard_Macro\saved_screenshots'


def load_files_func():
    screenshot_files = glob(join(SAVE_SCREENSHOTS_FLDR, "*.png"))
    hand_range_files = glob(join(SAVE_SCREENSHOTS_FLDR, "*.pkl"))
    # sort files by os.path.getmtime
    screenshot_files.sort(key=lambda x: os.path.getmtime(x))
    hand_range_files.sort(key=lambda x: os.path.getmtime(x))
    file_dict = {
        'extract_active_player_roi': [
            file for file in screenshot_files if 'ActivePlayerRoi' in file
        ],
        'crop_copy_nodes': [
            file for file in screenshot_files if 'CropCopyNodes' in file
        ],
        'hand_ranges': [
            file for file in hand_range_files if 'HandRanges' in file
        ],
    }
    return file_dict


def save_file_func(node_name, data, info, filetype="png", save_folder=SAVE_SCREENSHOTS_FLDR):
    """
    This function saves a copy of the screenshot of the screen or hand range data to a file.

    Args:
        node_name: The name of the node.
        data: The data to be saved (screenshot or hand range data).
        info: Additional information about the data.
        filetype: The type of the file to be saved (default: "png").
        save_folder: The folder where the files will be saved (default: SAVE_SCREENSHOTS_FLDR).

    Returns:
        None
    """
    count = len(glob(join(save_folder, "*.*")))
    count = f"{count:03}"
    dst = join(save_folder, f"{count}_{info}_{node_name}.{filetype}")

    if filetype == "png":
        cv2.imwrite(dst, data)
    elif filetype == "pkl":
        with open(dst, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported filetype '{filetype}'")



class CollectNodes:
    def __init__(
        self,
        save_screenshots: bool = False,
        test_mode: bool = False,
        load_files_func: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize the CollectNodes class.

        :param start_node: The starting node for the collection process.
        :param save_screenshots: If True, save screenshots during the collection process.
        :param test_mode: If True, use saved screenshots instead of taking new ones.
        """
        assert not (save_screenshots and test_mode), "save_screenshots and test_mode cannot both be True."

        self.node_name = "FirstAction"
        self.save_screenshots = save_screenshots
        self.test_mode = test_mode

        if test_mode:
            assert load_files_func is not None, "load_files_func must be provided when test_mode is True."
            self.test_files = load_files_func()

        # Start the process by collecting the first node - FirstAction
        logging.info("Starting node collection process\n\n")
        # Get the active player ROI and the screenshot of the screen
        active_roi, img_screenshot, self.test_files = ActivePlayerROI(self.node_name).extract_active_player_roi(save_screenshots, test_mode, self.test_files if test_mode else None)

        if self.save_screenshots:
            save_file_func(
                node_name=self.node_name,
                data=img_screenshot,
                info="ActivePlayerRoi",
                filetype="png")


        # Extract the action options from the screenshot
        extractor = ActionOptionsExtractor(img_screenshot, active_roi)
        # Store the OCR results in a dictionary
        ocr_dict = extractor.extract_action_options()
        # Copy the action ranges from the screen to the clipboard
        extraction = range_copier.copy_ranges(img_screenshot, ocr_dict, start_node)
        logging.info(f"Extraction collected: {extraction}\n\n")

        self.initialize_node_data(extraction)

    def initialize_node_data(self, extraction: List[Any]) -> None:
        """
        Initialize node data based on the provided extraction.

        :param extraction: A list containing node, position, hand_ranges_by_action,
            and node_branches.
        """
        node = extraction[0]
        logging.info(f"Node collected: {node}\n\n")
        position = extraction[1]
        logging.info(f"Position collected: {position}\n\n")
        hand_ranges_by_action = extraction[2]
        logging.info(f"Hand ranges collected: {hand_ranges_by_action}\n\n")
        node_branches = extraction[3]
        logging.info(f"Node branches collected: {node_branches}\n\n")

        new_nodes_l = poker_node_analyzer.new_nodes(node, position, node_branches)
        logging.info(f"New nodes collected: {new_nodes_l}\n\n")

        self.collected_node_info_l: List[List[Any]] = []
        self.collected_node_names_l: List[str] = []
        self.outstanding_nodes_info_to_collect_l: List[List[Any]] = []

        self.collected_node_info_l.append(extraction)
        self.collected_node_names_l.append(node)
        self.outstanding_nodes_info_to_collect_l.extend(iter(new_nodes_l))

    def check_for_reset(self) -> bool:
        """
        Check if a reset is needed based on the outstanding nodes and mouse position.

        :return: True if a reset is needed, False otherwise.
        """
        check_one = len(self.outstanding_nodes_info_to_collect_l) > 0
        check_two = pyautogui.position() != (0, 0)
        return all([check_one, check_two])

    def collect_node(self, reset=False) -> None:
        """
        Collect a node based on the current state.

        :param reset: If True, use the current node data instead of fetching a new one.
        """
        if not reset:
            node_data = self.outstanding_nodes_info_to_collect_l.pop(0)
            self.node_data = node_data
        else:
            node_data = self.node_data
        self.process_node_data(node_data)

    def process_node_data(self, node_data: List[Any]) -> None:
        """
        Process the given node data and update the internal state accordingly.

        :param node_data: A list containing node, x_mid, and y_mid.
        """
        node_name = node_data[0]
        if poker_node_analyzer.is_end_point_reached(node_name):
            print("End point reached")
            _ = poker_node_analyzer.click_node_start()
        else:
            self.collect_and_process_new_node(node_data)

    def collect_and_process_new_node(self, node_data: List[Any]) -> None:
        """
        Collect a new node based on the given node_data and update the internal state.

            :param node_data: A list containing node, x_mid, and y_mid.
        """
        x_mid = node_data[1]
        y_mid = node_data[2]
        pyautogui.moveTo(x_mid, y_mid)
        pyautogui.click()
        pyautogui.moveRel(0, 1000)
        sleep(1.2)
        if crown_detected := poker_node_analyzer.not_end_node():
            self.handle_non_end_node(node_data)
        else:
            _ = poker_node_analyzer.click_node_start()

    def handle_non_end_node(self, node_data: List[Any]) -> None:
        """
        Handle non-end node by extracting and processing new node data.

        :param node_data: A list containing node, x_mid, and y_mid.
        :param node_name: The name of the current node.
        """
        node_name = node_data[0]
        if poker_node_analyzer.is_end_point_reached(node_name):
            _ = poker_node_analyzer.click_node_start()
        else:
            active_roi, img_screenshot = ActivePlayerROI(self.node_name).extract_active_player_roi()
            extractor = ActionOptionsExtractor(img_screenshot, active_roi)
            ocr_dict = extractor.extract_action_options()
            logging.info(f"OCR dict: {ocr_dict}\n\n")
            position = ocr_dict["position"]

            if not poker_node_analyzer.ignore_node_check(node_name):
                self.extract_and_update_node_data(node_name, img_screenshot, ocr_dict)

    def extract_and_update_node_data(
        self, node_name: str, img_screenshot: np.ndarray, ocr_dict: Dict[str, Any]
    ) -> None:
        """
        Extract new node data, update the internal state, and collect new nodes.

        :param node_name: The name of the current node.
        :param img_screenshot: The screenshot image of the current node.
        :param ocr_dict: The OCR dictionary containing node information.
        """
        extraction = range_copier.copy_ranges(img_screenshot, ocr_dict, node_name)
        node_name = extraction[0]
        logging.info(f"Node: {node_name}\n\n")
        position = extraction[1]
        logging.info(f"Position: {position}\n\n")
        hand_ranges_by_action = extraction[2]
        logging.info(f"Hand ranges by action: {hand_ranges_by_action}\n\n")
        node_branches = extraction[3]
        logging.info(f"Node branches: {node_branches}\n\n")

        self.collected_node_info_l.append(extraction)
        self.collected_node_names_l.append(node_name)
        new_nodes_l = poker_node_analyzer.new_nodes(node_name, position, node_branches)

        self.outstanding_nodes_info_to_collect_l = [
            node for node in new_nodes_l if node not in self.collected_node_names_l
        ] + self.outstanding_nodes_info_to_collect_l

        for x in new_nodes_l:
            self.outstanding_nodes_info_to_collect_l.append(x)

    def loop_collect(self) -> None:
        """
        Continuously collect nodes until a reset is required.
        """
        while self.check_for_reset():
            _ = self.collect_node()
            logging.info(f'Outstanding nodes to collect: {self.outstanding_nodes_info_to_collect_l}\n\n')

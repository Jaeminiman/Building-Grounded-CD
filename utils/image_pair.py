import cv2
import os
import numpy as np
from typing import Tuple

class ImagePair:
    def __init__(self, image1_path: str, image2_path: str):
        """
        Initialize the ImagePair class with two image paths.

        Args:
            image1_path (str): Path to the first image (initial).
            image2_path (str): Path to the second image (new).
        """
        # Set image paths
        self.image1_path = image1_path
        self.image2_path = image2_path

        # File name alignment restriction
        assert(os.path.splitext(os.path.basename(self.image1_path))[0] == os.path.splitext(os.path.basename(self.image2_path))[0])

        # Initialize image data
        self.image1 = None
        self.image2 = None

        # Get filename
        self.filename = os.path.splitext(os.path.basename(self.image1_path))[0]

        # Load images
        self._load_images()

    def _load_images(self) -> None:
        """
        Load images from provided paths.

        Raises:
            FileNotFoundError: If one or both image paths are invalid.
        """
        if os.path.exists(self.image1_path) and os.path.exists(self.image2_path):
            self.image1 = cv2.imread(self.image1_path)
            self.image2 = cv2.imread(self.image2_path)
        else:
            raise FileNotFoundError("One or both image paths are invalid.")

    def get_image_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image pair as numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The loaded images as numpy arrays.
        """
        return self.image1, self.image2

    def get_image_shapes(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get the shapes of the loaded images.

        Returns:
            Tuple[Tuple[int, int, int], Tuple[int, int, int]]: The shapes of the two images.

        Raises:
            ValueError: If the images are not loaded properly.
        """
        if self.image1 is not None and self.image2 is not None:
            return self.image1.shape, self.image2.shape
        else:
            raise ValueError("Images are not loaded properly.")

    def reload_images(self) -> None:
        """
        Reload images from the given paths.
        """
        self._load_images()

    def save_images(self, output_dir: str) -> None:
        """
        Save the images to the specified output directory.

        Args:
            output_dir (str): Path to the output directory.

        Raises:
            ValueError: If the images are not loaded properly.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define output paths
        image1_output_path = os.path.join(output_dir, 'initial', f'{self.filename}.jpg')
        image2_output_path = os.path.join(output_dir, 'new', f'{self.filename}.jpg')

        # Save images
        if self.image1 is not None and self.image2 is not None:
            os.makedirs(os.path.dirname(image1_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(image2_output_path), exist_ok=True)
            cv2.imwrite(image1_output_path, self.image1)
            cv2.imwrite(image2_output_path, self.image2)
        else:
            raise ValueError("Images are not loaded properly.")

    def rectify(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rectify the initial image to align with the new image using homography.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                The corrected initial image, corrected new image, and the homography matrix.
        """
        # Copy the original images
        image1 = self.image1.copy()  # initial
        image2 = self.image2.copy()  # new

        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # Create SIFT feature detector
        sift = cv2.SIFT_create()

        # Detect keypoints and descriptors in both images
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Create BFMatcher and match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test to select good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract points from the good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Compute homography using RANSAC
        homography_12, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Warp the initial image to align with the new image
        height, width, channel = image2.shape
        corrected_image1 = cv2.warpPerspective(image1, homography_12, (width, height))

        # Create a mask to mark valid pixel areas in the warped image
        mask_warped = np.zeros_like(image1, dtype=np.uint8)
        cv2.warpPerspective(np.ones_like(image1, dtype=np.uint8), homography_12, (width, height), dst=mask_warped, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Apply mask to the new image
        corrected_image2 = image2 * mask_warped

        return corrected_image1, corrected_image2, homography_12


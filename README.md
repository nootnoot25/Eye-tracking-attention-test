# Color Matching Test with Gaze Estimation

This Python program uses OpenCV, Pygame, Mediapipe, and Numpy to conduct a color matching test with gaze estimation. The program presents blue and red squares on the screen and requires the user to focus on them based on gaze direction, detected using a webcam and Mediapipe's face mesh. This test is based on continuous performance tests used for ADHD screening

## Requirements

- Python 3.x
- Pygame
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- Numpy

## Installation

1. Clone the repository:
   git clone https://github.com/nootnoot25/Eye-tracking-attention-test
2. Install dependencies:
   pip install pygame opencv-python mediapipe numpy


## Usage

1. Run the program:
   python color_matching_test.py
2. Follow the on-screen instructions to calibrate and perform the color matching test.
3. Press `Q` or close the window to exit the test.

## How It Works

- **Gaze Estimation:** Uses Mediapipe's face mesh to estimate gaze direction based on eye landmarks.
- **Calibration:** Calibrates the user's head pose and gaze direction before the test begins.
- **Test Execution:** Presents colored squares and tracks user gaze to determine if they correctly look at the squares.
- **Results:** Prints test results including counts of blue and red squares, correct looks, correct presses, average reaction time, and non-target presses.
  

## Authors

Mohamed Yasin Azeez



import pygame
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import sys
from collections import deque


class GazeEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                    min_detection_confidence=0.5)

    def estimate_gaze_and_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        left_gaze_vector = None
        right_gaze_vector = None
        head_pose = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye_landmarks = [landmarks[159], landmarks[145], landmarks[155]]
            right_eye_landmarks = [landmarks[386], landmarks[374], landmarks[380]]
            left_gaze_vector = np.array([left_eye_landmarks[1].x - left_eye_landmarks[0].x,
                                         left_eye_landmarks[1].y - left_eye_landmarks[0].y,
                                         left_eye_landmarks[1].z - left_eye_landmarks[0].z])
            right_gaze_vector = np.array([right_eye_landmarks[1].x - right_eye_landmarks[0].x,
                                          right_eye_landmarks[1].y - right_eye_landmarks[0].y,
                                          right_eye_landmarks[1].z - right_eye_landmarks[0].z])

            # Estimate head pose
            nose_landmark = landmarks[3]
            head_pose = (nose_landmark.x, nose_landmark.y, nose_landmark.z)

        return left_gaze_vector, right_gaze_vector, head_pose


class TestResultsPrinter:
    def __init__(self, blue_square_count, red_square_count, correct_look_count, correct_press_count, avg_reaction_time,
                 non_target_press_count):
        self.blue_square_count = blue_square_count
        self.red_square_count = red_square_count
        self.correct_look_count = correct_look_count
        self.correct_press_count = correct_press_count
        self.avg_reaction_time = avg_reaction_time
        self.non_target_press_count = non_target_press_count

    def print_results(self):
        print("Test results:")
        print("Blue squares generated:", self.blue_square_count)
        print("Red squares generated:", self.red_square_count)
        print("Correct looks:", self.correct_look_count)
        print("Correct presses:", self.correct_press_count)
        print("Average reaction time:", self.avg_reaction_time)
        print("Number of times spacebar pressed without target:", self.non_target_press_count)


class ColorMatchingTest:
    def __init__(self):
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.last_draw_time = time.time()
        self.square_visible = False
        self.square_color = None
        self.screen_width = 0
        self.screen_height = 0
        self.look_start_time = None
        self.look_duration_threshold = 0.3  # in seconds
        self.gaze_estimator = GazeEstimator()  # Create an instance of GazeEstimator
        self.calibrated_head_pose = None
        self.head_pose_deviation_threshold = 0.8
        self.blue_square_count = 0
        self.red_square_count = 0
        self.correct_look_count = 0
        self.correct_press_count = 0
        self.reaction_times = []
        self.non_target_press_count = 0

    def calibrate(self, cap, screen):
        calibration_time = 5  # Duration of calibration in seconds
        calibration_start_time = time.time()
        left_head_poses = []
        right_head_poses = []

        def display_text(text):
            screen.fill(self.WHITE)
            text_surface = pygame.font.SysFont(None, 36).render(text, True, (0, 0, 0))
            screen.blit(text_surface, (self.screen_width // 2 - text_surface.get_width() // 2,
                                       self.screen_height // 2 + text_surface.get_height()))
            pygame.display.flip()

        def record_head_poses():
            nonlocal left_head_poses, right_head_poses
            _, _, z = head_pose
            if self.calibrated_head_pose is not None:
                # Calculate the deviation from calibrated head pose
                deviation = np.linalg.norm(np.array(head_pose) - np.array(self.calibrated_head_pose))
                if deviation < self.head_pose_deviation_threshold:
                    if gaze_direction == "Left":
                        left_head_poses.append(z)
                    elif gaze_direction == "Right":
                        right_head_poses.append(z)

        # Step 1: Align head in the center
        display_text("Please align your head in the center of the camera")
        time.sleep(2)
        while time.time() - calibration_start_time < calibration_time+2:
            ret, frame = cap.read()
            if not ret:
                break

            left_gaze_vector, right_gaze_vector, head_pose = self.gaze_estimator.estimate_gaze_and_pose(frame)

            if left_gaze_vector is not None and right_gaze_vector is not None:
                angle_left = np.degrees(np.arccos(np.dot(left_gaze_vector, [0, 0, 1]) /
                                                  (np.linalg.norm(left_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                angle_right = np.degrees(np.arccos(np.dot(right_gaze_vector, [0, 0, 1]) /
                                                   (np.linalg.norm(right_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                if angle_left < 10 and angle_right < 10:
                    self.calibrated_head_pose = head_pose  # Remember the head pose during calibration
                    break

            # Display webcam feed
            frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirrored display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for pygame
            frame = np.rot90(frame)  # Rotate frame 90 degrees counterclockwise
            frame = pygame.surfarray.make_surface(frame)  # Convert frame to pygame surface

            screen.fill(self.WHITE)
            screen.blit(frame, (self.screen_width // 2 - frame.get_width() // 2,
                                self.screen_height // 2 - frame.get_height() // 2))  # Center the frame
            pygame.display.flip()

        # Step 2: Look left
        display_text("Focus on the LEFT blue square")
        while time.time() - calibration_start_time < 2 * calibration_time:
            ret, frame = cap.read()
            if not ret:
                break

            left_gaze_vector, right_gaze_vector, head_pose = self.gaze_estimator.estimate_gaze_and_pose(frame)

            if left_gaze_vector is not None and right_gaze_vector is not None:
                angle_left = np.degrees(np.arccos(np.dot(left_gaze_vector, [0, 0, 1]) /
                                                  (np.linalg.norm(left_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                angle_right = np.degrees(np.arccos(np.dot(right_gaze_vector, [0, 0, 1]) /
                                                   (np.linalg.norm(right_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                if angle_left < 10 and angle_right < 10:
                    gaze_direction = "Left"
                    record_head_poses()

            # Display example square
            self.draw_square(screen, self.BLUE, "left")

            pygame.display.flip()

        # Step 3: Look right
        display_text("Focus on the RIGHT blue square")
        while time.time() - calibration_start_time < 3 * calibration_time:
            ret, frame = cap.read()
            if not ret:
                break

            left_gaze_vector, right_gaze_vector, head_pose = self.gaze_estimator.estimate_gaze_and_pose(frame)

            if left_gaze_vector is not None and right_gaze_vector is not None:
                angle_left = np.degrees(np.arccos(np.dot(left_gaze_vector, [0, 0, 1]) /
                                                  (np.linalg.norm(left_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                angle_right = np.degrees(np.arccos(np.dot(right_gaze_vector, [0, 0, 1]) /
                                                   (np.linalg.norm(right_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                if angle_left < 10 and angle_right < 10:
                    gaze_direction = "Right"
                    record_head_poses()

            # Display example square
            self.draw_square(screen, self.BLUE, "right")

            pygame.display.flip()

        # Calculate head pose ranges and thresholds
        if left_head_poses and right_head_poses:
            min_left = min(left_head_poses)
            max_right = max(right_head_poses)
            self.calibrated_head_pose = (
                self.calibrated_head_pose[0], self.calibrated_head_pose[1], (min_left + max_right) / 2)
            self.head_pose_deviation_threshold = abs(max_right - min_left) / 2

    def open_fullscreen_window(self):
        pygame.init()
        screen_info = pygame.display.Info()
        self.screen_width = screen_info.current_w
        self.screen_height = screen_info.current_h
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def end_test(self):
        avg_reaction_time = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0
        results_printer = TestResultsPrinter(self.blue_square_count, self.red_square_count,
                                             self.correct_look_count, self.correct_press_count,
                                             avg_reaction_time, self.non_target_press_count)
        results_printer.print_results()
        pygame.quit()
        sys.exit()

    def draw_square(self, screen, color, position):
        square_size = 50
        if position == "left":
            square_rect = pygame.Rect(10, self.screen_height // 2 - square_size // 2, square_size, square_size)
        elif position == "right":
            square_rect = pygame.Rect(self.screen_width - 10 - square_size,
                                      self.screen_height // 2 - square_size // 2,
                                      square_size, square_size)
        elif position == "center":
            square_rect = pygame.Rect(self.screen_width // 2 - square_size // 2,
                                      self.screen_height // 2 - square_size // 2,
                                      square_size, square_size)
        pygame.draw.rect(screen, color, square_rect)

    def generate_next_square_color(self):
        target_ratio = 3.5
        non_target_ratio = 1 / target_ratio
        if self.square_color == self.BLUE:
            self.red_square_count += 1  # Increment red square count when blue square is generated
            return random.choices([self.BLUE, self.RED], weights=[non_target_ratio, target_ratio])[0]
        else:
            self.blue_square_count += 1  # Increment blue square count when red square is generated
            return random.choices([self.BLUE, self.RED], weights=[target_ratio, non_target_ratio])[0]

    def generate_next_square_position(self):
        return random.choice(["left", "right"])

    def display_countdown(self, screen):
        countdown_duration = 3
        for i in range(countdown_duration, 0, -1):
            screen.fill(self.WHITE)
            text_surface = pygame.font.SysFont(None, 72).render(str(i), True, (0, 0, 0))
            screen.blit(text_surface, (self.screen_width // 2 - text_surface.get_width() // 2,
                                        self.screen_height // 2 - text_surface.get_height() // 2))
            pygame.display.flip()
            time.sleep(1)

    def run_test(self):
        screen = self.open_fullscreen_window()
        cap = cv2.VideoCapture(0)
        buffer_size = 30
        angle_buffer = deque(maxlen=buffer_size)
        threshold_angle = 79
        start_time = time.time()
        correct_look = False
        correct_press = False

        correct_press_timestamps = []

        self.calibrate(cap, screen)

        self.display_countdown(screen)  # Display countdown after calibration

        test_duration = 5 * 60  # 5 minutes in seconds
        end_time = time.time() + test_duration

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            left_gaze_vector, right_gaze_vector, head_pose = self.gaze_estimator.estimate_gaze_and_pose(frame)

            if left_gaze_vector is not None and right_gaze_vector is not None:
                angle_left = np.degrees(np.arccos(np.dot(left_gaze_vector, [0, 0, 1]) /
                                                  (np.linalg.norm(left_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                angle_right = np.degrees(np.arccos(np.dot(right_gaze_vector, [0, 0, 1]) /
                                                   (np.linalg.norm(right_gaze_vector) * np.linalg.norm([0, 0, 1]))))
                angle_buffer.append((angle_left + angle_right) / 2)

                if head_pose is not None:
                    _, _, z = head_pose
                    if self.calibrated_head_pose is not None:
                        # Calculate the deviation from calibrated head pose
                        deviation = np.linalg.norm(np.array(head_pose) - np.array(self.calibrated_head_pose))
                        if deviation > self.head_pose_deviation_threshold:  # Suppress left and right signals if deviation is significant
                            angle_buffer.clear()  # Clear the angle buffer to prevent false positives
                            continue

                if time.time() - start_time >= 1:
                    threshold_angle = sum(angle_buffer) / len(angle_buffer)
                    start_time = time.time()

                if angle_left < threshold_angle and angle_right < threshold_angle:
                    gaze_direction = "Straight"
                elif angle_left < threshold_angle:
                    gaze_direction = "Left"
                elif angle_right < threshold_angle:
                    gaze_direction = "Right"
                else:
                    gaze_direction = "Straight Ahead"

                if self.square_visible and gaze_direction == "Straight":
                    if self.square_color == self.BLUE:
                        if self.look_start_time is None:
                            self.look_start_time = time.time()  # Start timer when looking at blue square
                        elif time.time() - self.look_start_time >= self.look_duration_threshold:
                            self.square_visible = False
                            correct_look = True

                else:
                    angle_buffer.clear()
                    self.look_start_time = None

            current_time = time.time()
            if current_time - self.last_draw_time > 3:
                self.square_visible = True
                self.last_draw_time = current_time
                self.square_color = self.generate_next_square_color()
                square_start_time = current_time
                square_position = self.generate_next_square_position()

            if self.square_visible and current_time - square_start_time > 2:
                self.square_visible = False
                angle_buffer.clear()  # Clear angle buffer when the blue square disappears

            screen.fill(self.WHITE)
            if self.square_visible:
                self.draw_square(screen, self.square_color, square_position)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_test()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.end_test()
                    elif event.key == pygame.K_SPACE:
                        if self.square_color == self.RED:
                            self.square_visible = False
                            correct_press = True
                            self.correct_press_count += 1
                            correct_press_timestamps.append(time.time())
                        else:
                            self.non_target_press_count += 1

            if correct_look or correct_press:
                if correct_press:
                    reaction_time = correct_press_timestamps[-1] - square_start_time
                    self.reaction_times.append(reaction_time)
                screen.fill(self.WHITE)
                self.draw_square(screen, self.GREEN, "center")
                pygame.display.flip()
                time.sleep(2)
                self.correct_look_count += 1 if correct_look else 0  # Increment correct look count if correct_look is True
                correct_look = False
                correct_press = False

            if time.time() >= end_time:
                self.end_test()
                break


if __name__ == "__main__":
    test = ColorMatchingTest()
    test.run_test()

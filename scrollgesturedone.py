"""Improved gesture-controlled virtual mouse.

Refactored for clarity and testability: the procedural code has been wrapped
into a GestureController class that maintains state instead of using globals.
Added type hints, logging, and safer error handling.
"""

from __future__ import annotations

import os
import random
import time
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Controller, Button

import util

# Configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

mpHands = mp.solutions.hands

# Default media-pipe hands configuration
_MP_HANDS_CONFIG = dict(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1,
)


@dataclass
class GestureConfig:
    thumb_threshold: float = 60.0
    click_cooldown: float = 0.45
    palm_closed_threshold: float = 70.0
    palm_open_threshold: float = 120.0
    stop_dist_threshold: float = 6.0
    stop_time_threshold: float = 0.25
    half_fold_low: float = 45.0
    half_fold_high: float = 95.0
    scroll_sensitivity: float = 0.55
    scroll_smooth_alpha: float = 0.72
    # Angle threshold to detect the thumb as 'open' (degrees)
    thumb_angle_threshold: float = 50.0


class GestureController:
    """Encapsulates detection and action state for gesture-driven mouse control.

    Example:
        ctrl = GestureController()
        ctrl.run()
    """

    def __init__(self, config: GestureConfig | None = None, camera_index: int = 0):
        self.config = config or GestureConfig()
        self.mouse = Controller()
        self.hands = mpHands.Hands(**_MP_HANDS_CONFIG)
        self.camera_index = camera_index

        # internal state (replaces previous global variables)
        self._last_click_time = 0.0
        self._palm_was_closed = False

        self._last_index_pos: Tuple[int, int] = (0, 0)
        self._stop_start_time = 0.0

        self._scroll_mode = False
        self._last_scroll_y = 0
        self._scroll_vel = 0.0
        self._scroll_accum = 0.0
        self._scroll_gesture_state = "idle"  # idle, folded, open, ready
        self._scroll_gesture_time = 0.0
        # When True the main loop should exit (set by Thumb+Pinky gesture)
        self._should_exit = False

    # -- Gesture helpers -------------------------------------------------
    @staticmethod
    def _index_tip_from_processed(processed) -> Optional[mp.framework.formats.landmark_pb2.NormalizedLandmark]:
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return None

    def _move_mouse(self, index_finger_tip) -> None:
        if index_finger_tip is None:
            return
        x = int(index_finger_tip.x * SCREEN_WIDTH)
        y = int(index_finger_tip.y * SCREEN_HEIGHT)
        try:
            self.mouse.position = (x, y)
        except Exception:
            logger.exception("Failed to move mouse")

    # click/gesture predicates (kept as simple logic wrappers)
    @staticmethod
    def _is_left_click(landmark_list: Sequence[Tuple[float, float]], thumb_index_dist: float) -> bool:
        return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
            and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
            and thumb_index_dist > 50
        )

    @staticmethod
    def _is_right_click(landmark_list: Sequence[Tuple[float, float]], thumb_index_dist: float) -> bool:
        return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
            and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
            and thumb_index_dist > 50
        )

    @staticmethod
    def _is_double_click(landmark_list: Sequence[Tuple[float, float]], thumb_index_dist: float) -> bool:
        return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
            and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
            and thumb_index_dist > 50
        )

    def _take_screenshot(self) -> Optional[str]:
        try:
            im = pyautogui.screenshot()
            label = random.randint(1, 1000)
            screenshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            file_path = os.path.join(screenshots_dir, f"screenshot_{label}.png")
            im.save(file_path)
            logger.info("Screenshot saved to %s", file_path)
            return file_path
        except Exception:
            logger.exception("Failed to take screenshot")
            return None

    # -- Main per-frame processing ---------------------------------------
    def detect_gesture(self, frame: 'ndarray', landmark_list: List[Tuple[float, float]], processed) -> None:  # type: ignore[valid-type]
        """Process a camera frame and potentially perform actions (move, click, scroll).

        This method mirrors the original behavior but keeps state in the instance.
        """
        now = time.time()
        if len(landmark_list) < 21:
            return

        index_finger_tip = self._index_tip_from_processed(processed)
        frame_h, frame_w = frame.shape[:2]
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]], scale=(frame_w, frame_h))

        cfg = self.config

        # palm openness (average fingertip-to-wrist distance)
        try:
            wrist = landmark_list[0]
            tip_indices = [4, 8, 12, 16, 20]
            dists = [util.get_distance([landmark_list[i], wrist], scale=(frame_w, frame_h)) for i in tip_indices]
            avg_tip_wrist = sum(dists) / len(dists)
        except Exception:
            avg_tip_wrist = 0.0

        # Debug overlays
        cv2.putText(frame, f"thumb_dist:{thumb_index_dist:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"palm_avg:{avg_tip_wrist:.1f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"palm_state:{'closed' if self._palm_was_closed else 'open'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # palm close->open screenshot
        if avg_tip_wrist < cfg.palm_closed_threshold:
            self._palm_was_closed = True
        elif avg_tip_wrist > cfg.palm_open_threshold and self._palm_was_closed and (now - self._last_click_time) > cfg.click_cooldown:
            file_path = self._take_screenshot()
            if file_path:
                file_name = os.path.basename(file_path)
                # Log and print the filename and full path to ensure visibility in the terminal
                logger.info("Screenshot saved: %s", file_name)
                logger.info("Screenshot full path: %s", file_path)
                print(f"Screenshot saved: {file_name}")
                sys.stdout.flush()
                cv2.putText(frame, f"Saved: {file_name}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Screenshot failed", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self._last_click_time = now
            self._palm_was_closed = False

        # movement vs clicking
        current_index_screen = (int(landmark_list[8][0] * SCREEN_WIDTH), int(landmark_list[8][1] * SCREEN_HEIGHT))

        # initialize last index pos on first valid frame
        if self._last_index_pos == (0, 0):
            self._last_index_pos = current_index_screen
            self._stop_start_time = 0.0
            self._scroll_mode = False
            self._last_scroll_y = current_index_screen[1]

        dx = current_index_screen[0] - self._last_index_pos[0]
        dy = current_index_screen[1] - self._last_index_pos[1]
        dist_moved = (dx * dx + dy * dy) ** 0.5

        stopped = False
        if dist_moved < cfg.stop_dist_threshold:
            if self._stop_start_time == 0:
                self._stop_start_time = now
            elif (now - self._stop_start_time) > cfg.stop_time_threshold:
                stopped = True
        else:
            self._stop_start_time = 0.0

        self._last_index_pos = current_index_screen

        # -- Simplified Scroll Logic --
        # Condition: Thumb not pinched (Move Mode) AND Index & Middle fingers Extended.
        index_angle = util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8])
        middle_angle = util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12])

        cv2.putText(frame, f"idx_ang:{index_angle:.1f} mid_ang:{middle_angle:.1f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"thumb:{thumb_index_dist:.1f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check if we should be in scroll mode
        # User-request: Only enable scrolling when the Index and Middle fingers are FOLDED.
        # Folded is indicated by smaller joint angles; use the configured half_fold_low as threshold.
        is_scroll_posture = (index_angle < cfg.half_fold_low and middle_angle < cfg.half_fold_low)
        
        # Compute pinky state early so it can act as an explicit "close scroll" gesture
        pinky_angle = util.get_angle(landmark_list[17], landmark_list[18], landmark_list[20])
        pinky_open = pinky_angle > 95
        cv2.putText(frame, f"Pinky: {int(pinky_angle)} {'DOWN (Open)' if pinky_open else 'UP (Closed)'}", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Compute thumb openness more robustly using the thumb joint angles
        thumb_angle = util.get_angle(landmark_list[2], landmark_list[3], landmark_list[4])
        thumb_open = thumb_angle > cfg.thumb_angle_threshold
        cv2.putText(frame, f"thumb_ang:{thumb_angle:.1f} open:{int(thumb_open)}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Compute ring finger angle to ensure only thumb+pinky are extended
        ring_angle = util.get_angle(landmark_list[13], landmark_list[14], landmark_list[16])
        ring_folded = ring_angle < cfg.half_fold_low

        # Debug: show computed posture booleans and current scroll state
        cv2.putText(frame, f"is_scroll_posture:{int(is_scroll_posture)} thumb_dist:{int(thumb_index_dist)} scroll:{int(self._scroll_mode)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.putText(frame, f"ring_ang:{ring_angle:.1f} folded:{int(ring_folded)}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Special gesture: Thumb + Pinky -> exit program (require short hold to avoid accidental triggers)
        # Require BOTH thumb_open and pinky_open AND other fingers folded (prevents full-palm-open from triggering exit)
        exit_posture = bool(thumb_open and pinky_open and index_angle < cfg.half_fold_low and middle_angle < cfg.half_fold_low and ring_folded)
        cv2.putText(frame, f"exit_posture:{int(exit_posture)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        if exit_posture:
            if self._scroll_gesture_time == 0.0:
                self._scroll_gesture_time = now
            elif (now - self._scroll_gesture_time) > 0.15:  # 150 ms hold
                # Trigger program exit
                self._should_exit = True
                logger.info("Exit requested by Thumb+Pinky gesture (strict posture)")
                cv2.putText(frame, "Exit requested (Thumb+Pinky)", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # reset timer after action
                self._scroll_gesture_time = 0.0
        else:
            # reset debounce timer when gesture not present
            self._scroll_gesture_time = 0.0

            if thumb_index_dist >= cfg.thumb_threshold:
                if is_scroll_posture:
                    if not self._scroll_mode:
                        # Entered scroll mode this frame
                        self._scroll_mode = True
                        self._last_scroll_y = current_index_screen[1]
                        self._scroll_accum = 0.0
                        logger.info("Scroll mode entered (Index+Middle folded)")
                    
                    cv2.putText(frame, "Scroll Mode ACTIVE", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
                else:
                    self._scroll_mode = False
            else:
                # Pinching (Move Mode) exits scroll mode
                self._scroll_mode = False

        # Execute Scroll
        # Execute Scroll
        if self._scroll_mode:
            # -- Direction Lock Logic --
            # User Request: Pinky Open = Scroll Down. Pinky Closed = Scroll Up.
            # Speed is determined by how fast hand moves (absolute distance), regardless of direction.
            
            scroll_dy = current_index_screen[1] - self._last_scroll_y
            move_magnitude = abs(scroll_dy)
            
            # Use earlier computed pinky state (thumb+pinky closure gesture handled above)
            p_state = "DOWN (Open)" if pinky_open else "UP (Closed)"
            # Display Pinky State for Debug
            cv2.putText(frame, f"Pinky: {int(pinky_angle)} {p_state}", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 1. Deadzone
            if move_magnitude < 5:
                move_magnitude = 0.0
            
            # 2. Sensitivity & Direction
            # Positive Scroll = UP. Negative Scroll = DOWN.
            factor = 6.0
            
            if pinky_open:
                # Force Scroll DOWN (Negative)
                target_scroll = -1 * move_magnitude * factor
            else:
                # Force Scroll UP (Positive)
                target_scroll = 1 * move_magnitude * factor

            # Simple smoothing
            self._scroll_vel = cfg.scroll_smooth_alpha * self._scroll_vel + (1 - cfg.scroll_smooth_alpha) * target_scroll
            self._scroll_accum += self._scroll_vel

            # 3. Accumulate
            send = int(self._scroll_accum)
            
            if abs(send) >= 20: 
                try:
                    pyautogui.scroll(send)
                    self._scroll_accum -= send
                except Exception:
                    logger.exception("Scroll failed")
            
            self._last_scroll_y = current_index_screen[1]
            
            cv2.putText(frame, f"Scroll: {send} ({p_state})", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # movement vs clicks
        if self._scroll_mode:
            pass
        elif thumb_index_dist < cfg.thumb_threshold:
            self._move_mouse(index_finger_tip)
        else:
            if self._is_left_click(landmark_list, thumb_index_dist) and (now - self._last_click_time) > cfg.click_cooldown:
                self.mouse.press(Button.left)
                self.mouse.release(Button.left)
                self._last_click_time = now
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self._is_right_click(landmark_list, thumb_index_dist) and (now - self._last_click_time) > cfg.click_cooldown:
                self.mouse.press(Button.right)
                self.mouse.release(Button.right)
                self._last_click_time = now
                cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self._is_double_click(landmark_list, thumb_index_dist) and (now - self._last_click_time) > cfg.click_cooldown:
                try:
                    pyautogui.doubleClick()
                except Exception:
                    logger.exception("Double click failed")
                self._last_click_time = now
                cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -- Run loop --------------------------------------------------------
    def run(self, show_window: bool = True) -> None:
        """Start camera loop and process frames until 'q' is pressed."""
        draw = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = self.hands.process(frame_rgb)

                landmark_list: List[Tuple[float, float]] = []
                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        landmark_list.append((lm.x, lm.y))

                try:
                    self.detect_gesture(frame, landmark_list, processed)
                except Exception:
                    logger.exception("Error during gesture detection; continuing loop")

                # If the Thumb+Pinky gesture requested exit, break the main loop (keeps 'q' to quit as well)
                if self._should_exit:
                    logger.info("Exiting main loop due to Thumb+Pinky gesture")
                    break

                if show_window:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    ctrl = GestureController()
    try:
        ctrl.run()
    except KeyboardInterrupt:
        logger.info("Exiting on user request")
    except Exception:
        logger.exception("Unhandled error in main")


if __name__ == "__main__":
    main()


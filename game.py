import cv2
import random
import time
import mediapipe as mp
import numpy as np
from AppKit import NSScreen

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

screen = NSScreen.mainScreen().frame()
screen_width = int(screen.size.width)
screen_height = int(screen.size.height)

cv2.namedWindow("Ball Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Ball Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize mediapipe face mesh detector
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Ball properties
ball_radius = 40
ball_color = (0, 0, 255)  # Red color
balls = []
max_balls = 5  # Initial limit on balls
ball_interval = 1.0  # Initial time interval (seconds) between new ball drops
last_ball_time = 0
base_speed = 5  # Initial ball speed

# Game properties
lives = 3
score = 0
best_score = 0  # Track best score across all rounds
start_time = time.time()  # Track game start time

# Screen dimensions (to avoid edge drops)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
safe_margin = ball_radius + 50  # Avoid edges for ball spawning

# Class to represent the falling balls
class Ball:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def update_position(self):
        self.y += self.speed  # Move the ball down

# --- Replace your draw_text_block() with this improved UI version ---
def draw_text_block(frame, text_lines, center_y):
    """
    Draws a centered instructional text box with smooth UI spacing.
    text_lines = [("text", scale), ...]
    """
    padding_x = 50
    padding_y = 20
    line_spacing = 12

    # Measure text block size
    widths = []
    heights = []
    for text, scale in text_lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        widths.append(w)
        heights.append(h)

    block_width = max(widths) + padding_x * 2
    block_height = sum(heights) + padding_y * 2 + line_spacing * (len(text_lines) - 1)

    # Center block
    x1 = frame_width // 2 - block_width // 2
    y1 = center_y - block_height // 2
    x2 = x1 + block_width
    y2 = y1 + block_height

    # Draw background (semi-transparent black)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Draw text lines centered
    y = y1 + padding_y
    for (text, scale), h in zip(text_lines, heights):
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        x = frame_width // 2 - w // 2
        cv2.putText(frame, text, (x, y + h), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2)
        y += h + line_spacing

    return frame


# --- Replace your show_waiting_screen() entirely with this ---
def show_waiting_screen():
    """Show camera feed + live mouth hitbox + instruction UI"""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Live mouth hitbox preview
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pts = [
                    face_landmarks.landmark[2],
                    face_landmarks.landmark[152],
                    face_landmarks.landmark[234],
                    face_landmarks.landmark[454]
                ]

                face_width = abs(pts[2].x - pts[3].x) * frame_width
                face_height = abs(pts[1].y - pts[0].y) * frame_height
                square_size = int(max(face_width, face_height) * 0.6)

                top_y = int(pts[0].y * frame_height - square_size / 7)
                bottom_y = int(pts[1].y * frame_height - square_size / 7)
                min_x = int(min(p.x for p in pts) * frame_width)
                max_x = int(max(p.x for p in pts) * frame_width)

                top_y = max(0, top_y)
                bottom_y = min(frame_height, bottom_y)
                min_x = max(safe_margin, min_x)
                max_x = min(frame_width - safe_margin, max_x)

                cv2.rectangle(frame, (min_x, top_y), (max_x, bottom_y), (0, 255, 0), 2)

        # Draw UI text
        frame = draw_text_block(frame, [
            ("Align your mouth box & move your head to control!", 0.9),
            ("Press ENTER to Start", 1.2),
            ("Press Q to Quit", 1.0),
        ], frame_height // 4)

        # Scale to full screen size
        frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Ball Game", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return ord('q')
        if key == 13:
            return 13



def show_game_over_screen():
    """Show camera feed with score and retry options"""
    global best_score

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)

        frame = draw_text_block(frame, [
            (f"GAME OVER!", 1.4),
            (f"Your Score: {score}", 1.2),
            (f"Best Score: {best_score}", 1.2),
            ("Press ENTER to Retry", 1.0),
            ("Press Q to Quit", 1.0),
        ], frame_height // 4)

        # Scale to full screen size
        frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Ball Game", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return ord('q')
        if key == 13:  # Enter
            return 13

# Main game loop
while True:
    # Show waiting screen before starting the game
    key = show_waiting_screen()
    if key == ord('q'):
        print("Exiting...")
        break  # Exit if 'q' is pressed

    # Reset game state
    lives = 1
    score = 0
    balls.clear()
    start_time = time.time()

    while lives > 0:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Mirror the frame for natural movement
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks using Face Mesh
        results = face_mesh.process(rgb_frame)

        # Adjust difficulty based on elapsed time
        elapsed_time = time.time() - start_time

        # Gradual increase in difficulty
        if elapsed_time > 5:
            max_balls = min(10, 1 + int(elapsed_time // 5))  # Gradual increase in number of balls, max 10
            ball_interval = max(0.3, 1.0 - (elapsed_time // 10) * 0.1)  # More aggressive reduction in interval
            base_speed = min(20, 1 + int(elapsed_time // 5))  # More aggressive speed increase

        # Add new balls at intervals
        current_time = time.time()
        if len(balls) < max_balls and (current_time - last_ball_time) > ball_interval:
            ball_x = random.randint(safe_margin, frame_width - safe_margin)  # Spawn away from edges
            ball_y = 0  # Start at the top
            ball_speed = random.randint(base_speed, base_speed + 10)  # Gradual speed increase
            balls.append(Ball(ball_x, ball_y, ball_speed))
            last_ball_time = current_time

        # Update ball positions
        for ball in balls[:]:  # Use a copy of the list to iterate safely
            ball.update_position()

            # Check if ball hits the ground
            if ball.y - ball_radius > frame_height:
                balls.remove(ball)
                lives -= 1  # Lose a life if ball touches the ground
                continue

            # Check for collision with tracked face
            caught = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Use a smaller bounding box that is half the size of the face
                    face_landmarks_list = [
                        face_landmarks.landmark[2],   # Nose tip (upper bound of mouth box)
                        face_landmarks.landmark[152], # Chin (lower bound of mouth box)
                        face_landmarks.landmark[234], # Left cheek (side bound)
                        face_landmarks.landmark[454]  # Right cheek (side bound)
                    ]

                    # Adjust bounds to make sure the bounding box is square
                    face_width = abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * frame_width  # Cheek-to-cheek distance
                    face_height = abs(face_landmarks.landmark[152].y - face_landmarks.landmark[2].y) * frame_height  # Nose to chin distance

                    # Choose the larger dimension to make the bounding box square
                    square_size = int(max(face_width, face_height) * 0.6)  # Reduce the size by 40%

                    # Calculate top and bottom bounds for the square bounding box
                    top_y = int(face_landmarks.landmark[2].y * frame_height - square_size / 7)  # Top of the square
                    bottom_y = int(face_landmarks.landmark[152].y * frame_height - square_size / 7) # Bottom of the square

                    # Calculate left and right bounds for the square bounding box
                    min_x = int(min(landmark.x for landmark in face_landmarks_list) * frame_width) 
                    max_x = int(max(landmark.x for landmark in face_landmarks_list) * frame_width) 

                    # Adjust bounds to make sure the bounding box is square and within the frame
                    if bottom_y > frame_height:
                        bottom_y = frame_height
                    if top_y < 0:
                        top_y = 0
                    if min_x < safe_margin:
                        min_x = safe_margin
                    if max_x > frame_width - safe_margin:
                        max_x = frame_width - safe_margin

                    # Draw the bounding box (optional: uncomment to see the "hitbox") 
                    cv2.rectangle(frame, (min_x, top_y), (max_x, bottom_y), (0, 255, 0), 2)

                    # Check if the ball is inside the bounding box
                    for ball in balls:
                        if min_x < ball.x < max_x and top_y < ball.y < bottom_y:
                            caught = True
                            score += 1  # Increase score if caught
                            balls.remove(ball)  # Remove ball if caught

        # Draw balls on screen
        for ball in balls:
            cv2.circle(frame, (ball.x, ball.y), ball_radius, ball_color, -1)

        # Display the current score and lives
        cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Lives: {lives}", (frame_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame
        # Scale to full screen size
        frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Ball Game", frame)


        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Game over screen
    key = show_game_over_screen()
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == 13:  # Enter key
        best_score = max(best_score, score)  # Update best score
        base_speed = 5
        continue

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import pygame
import numpy as np
from keras.models import model_from_json
import sys
import threading
import time
import random
import math
import json
import os

# Initialize Pygame
# ADD THIS LINE BEFORE pygame.init() FOR BETTER SOUND STABILITY, ESPECIALLY ON MACOS
pygame.mixer.pre_init(44100, -16, 2, 1024) # Frequency, size, channels, buffer (adjust buffer if needed)
pygame.init()
pygame.mixer.init() # Initialize the mixer for sounds

# Default screen sizes
SCREEN_SIZES = {
    "Small": (800, 600),
    "Medium": (1024, 768),
    "Large": (1280, 720),
    "Fullscreen": (0, 0)  # Special case for fullscreen
}

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 0.8
JUMP_STRENGTH = -15
PLATFORM_SPEED = 6
PLAYER_SPEED = 3  # Increased player speed for a more dynamic feel

# Colors (Mario-like palette)
MARIO_SKY_BLUE = (92, 148, 252) # Brighter blue for sky
MARIO_GROUND_BROWN = (176, 128, 64) # Brown for ground blocks
MARIO_GROUND_TOP_GREEN = (0, 168, 0) # Green for top of ground blocks
WHITE = (255, 255, 255)
MARIO_RED = (252, 0, 0) # Mario's hat and shirt
MARIO_BLUE = (0, 0, 252) # Mario's overalls
MARIO_SKIN = (255, 204, 153) # Mario's skin color
MARIO_YELLOW = (255, 255, 0) # Coins/Stars
BLACK = (0, 0, 0)
GOOMBA_BROWN = (160, 82, 45) # Goomba body color
GOOMBA_FEET = (80, 40, 20) # Goomba feet color
KOOPA_GREEN = (0, 170, 0) # Koopa Troopa shell green
KOOPA_SKIN = (255, 220, 150) # Koopa Troopa skin color
BUTTON_HOVER = (100, 100, 100)  # Color for button hover state

class Button:
    """
    A simple button class for Pygame menus.
    Handles drawing, hover states, and click detection.
    """
    def __init__(self, x, y, width, height, text, font_object, text_color=WHITE, button_color=MARIO_GROUND_TOP_GREEN, hover_color=BUTTON_HOVER, pressed_color=(50,50,50)): # Added font_object and pressed_color
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font_object # Use the passed font object
        self.text_color = text_color
        self.button_color = button_color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.is_hovered = False
        self.is_pressed = False # New state for pressed

    def draw(self, screen):
        """
        Draws the button on the screen, including its background, border, and text.
        Changes color on hover and press.
        """
        current_color = self.button_color
        if self.is_pressed:
            current_color = self.pressed_color
        elif self.is_hovered:
            current_color = self.hover_color

        pygame.draw.rect(screen, current_color, self.rect, border_radius=10) #
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=10)  # Button border #

        text_surface = self.font.render(self.text, True, self.text_color) #
        text_rect = text_surface.get_rect(center=self.rect.center) #

        # Offset text slightly if pressed
        if self.is_pressed:
            text_rect.move_ip(2, 2)

        screen.blit(text_surface, text_rect) #

    def handle_event(self, event):
        """
        Handles Pygame events for the button.
        Updates hover and pressed states.
        Returns True if the button is clicked (MOUSEBUTTONUP while hovered and pressed).
        """
        clicked = False
        if event.type == pygame.MOUSEMOTION: #
            self.is_hovered = self.rect.collidepoint(event.pos) #
            if not self.is_hovered: # If mouse moves off while pressed, unpress
                 self.is_pressed = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.is_hovered: # Left mouse button
                self.is_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP: #
            if event.button == 1 and self.is_hovered and self.is_pressed: #
                clicked = True # Button was clicked #
            self.is_pressed = False # Reset pressed state on any mouse up
        return clicked

class Settings:
    """
    Manages game settings, including screen size and sound volume.
    Loads and saves settings to a JSON file.
    """
    def __init__(self):
        self.settings_file = "game_settings.json"
        self.current_size = "Medium"  # Default size
        self.master_volume = 0.5 # Default master volume (0.0 to 1.0)
        self.music_volume = 0.5 # Default music volume
        self.sfx_volume = 0.5 # Default sound effects volume
        self.load_settings()
        
    def load_settings(self):
        """Loads screen size and volume settings from file."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.current_size = settings.get('screen_size', 'Medium')
                    self.master_volume = settings.get('master_volume', 0.5)
                    self.music_volume = settings.get('music_volume', 0.5)
                    self.sfx_volume = settings.get('sfx_volume', 0.5)
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Saves current screen size and volume settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump({
                    'screen_size': self.current_size,
                    'master_volume': self.master_volume,
                    'music_volume': self.music_volume,
                    'sfx_volume': self.sfx_volume
                }, f, indent=4) # Use indent for readability
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def apply_screen_size(self, size_name):
        """
        Applies the selected screen size, updates global dimensions,
        and returns the new Pygame screen surface.
        """
        self.current_size = size_name
        self.save_settings()
        
        width, height = SCREEN_SIZES[size_name]
        
        if size_name == "Fullscreen":
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            # Get actual resolution in fullscreen mode
            width, height = screen.get_size()
        else:
            screen = pygame.display.set_mode((width, height))
        
        # Update global screen dimensions to reflect the new size
        global SCREEN_WIDTH, SCREEN_HEIGHT
        SCREEN_WIDTH = width
        SCREEN_HEIGHT = height
        
        return screen
    
    def set_master_volume(self, volume):
        self.master_volume = max(0.0, min(1.0, volume))
        self.save_settings()

    def set_music_volume(self, volume):
        self.music_volume = max(0.0, min(1.0, volume))
        self.save_settings()

    def set_sfx_volume(self, volume):
        self.sfx_volume = max(0.0, min(1.0, volume))
        self.save_settings()

class EmotionDetector:
    """
    Handles emotion detection using a pre-trained Keras model and OpenCV webcam.
    Provides a fallback to keyboard controls if the model or webcam is unavailable.
    """
    def __init__(self):
        self.current_emotion = 'neutral'
        self.face_detected = False # New: Tracks if a face is currently detected
        self.model = None
        self.webcam = None
        self.face_cascade = None
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                       4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.latest_display_frame = None  # For storing the frame to be displayed in Pygame
        self.frame_lock = threading.Lock() # For thread-safe access to latest_display_frame
        self.setup_model()
        self.setup_camera()
        
    def setup_model(self):
        """Loads the emotion detection model and face cascade classifier."""
        try:
            # Load emotion detection model architecture from JSON
            json_file = open("emotiondetector.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            
            # Load model weights from H5 file
            self.model.load_weights("emotiondetector.h5") 
            
            # Load Haar cascade for face detection
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_file)
            
            print("Emotion detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using keyboard controls instead...")
            self.model = None # Set model to None to indicate failure
    
    def setup_camera(self):
        """Initializes the webcam if the model was loaded successfully."""
        if self.model: # Only try to set up camera if model is available
            try:
                self.webcam = cv2.VideoCapture(0) # Try to open default camera (0)
                if not self.webcam.isOpened():
                    print("Camera not available, using keyboard controls")
                    self.webcam = None # Set webcam to None if it can't be opened
            except Exception as e:
                print(f"Camera setup failed: {e}, using keyboard controls")
                self.webcam = None
    
    def extract_features(self, image):
        """Preprocesses a grayscale image for model prediction."""
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1) # Reshape for model input
        return feature / 255.0 # Normalize pixel values
    
    def detect_emotion(self):
        """
        Captures a frame from the webcam, detects faces, predicts emotion,
        and updates `self.current_emotion`. Stores the processed frame for Pygame display.
        """
        if not self.model or not self.webcam:
            self.face_detected = False
            with self.frame_lock: # Ensure thread-safe update
                self.latest_display_frame = None
            return self.current_emotion
            
        processed_frame_for_display = None # Initialize here
        try:
            ret, frame = self.webcam.read()
            if not ret: 
                self.face_detected = False
                self.current_emotion = 'neutral'
                with self.frame_lock: # Ensure thread-safe update
                    self.latest_display_frame = None
                return self.current_emotion
            
            display_frame = frame.copy() # Work on a copy for drawing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) 
            
            if len(faces) > 0:
                self.face_detected = True 
                (x, y, w, h) = faces[0] 
                face_img = gray[y:y+h, x:x+w] 
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw on display_frame
                
                face_img_resized = cv2.resize(face_img, (48, 48)) 
                img_features = self.extract_features(face_img_resized)
                prediction = self.model.predict(img_features, verbose=0) 
                emotion_label = self.labels[prediction.argmax()] 
                
                cv2.putText(display_frame, emotion_label, (x-10, y-10), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                
                self.current_emotion = emotion_label 
            else:
                self.face_detected = False 
                self.current_emotion = 'neutral' 
            
            # Resize the (potentially annotated) frame for display
            processed_frame_for_display = cv2.resize(display_frame, (200, 150))

        except Exception as e:
            print(f"Error in emotion detection: {e}")
            self.face_detected = False 
            self.current_emotion = 'neutral' 
            processed_frame_for_display = None # Ensure it's None on error
        
        # Store the frame for Pygame to pick up, under lock
        with self.frame_lock:
            self.latest_display_frame = processed_frame_for_display
        
        return self.current_emotion

class Player:
    """
    Represents the player character (Mario-like).
    Handles movement, jumping, and drawing.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 40
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.jump_count = 0
        self.max_jumps = 2
        self.last_emotion = 'neutral'  # Track last emotion to prevent continuous jumping
        
        # New attributes for Grow Up skill
        self.is_grown_up = False
        self.grow_up_start_time = 0
        self.grow_up_duration = 10 # seconds
        self.last_grow_up_time = -30 # Initialize to allow immediate use
        self.grow_up_cooldown = 30 # seconds
        
        # Original dimensions
        self.original_width = 30
        self.original_height = 40
    
    def update(self, emotion, platforms, face_detected_status=True, game_sounds=None): # Added game_sounds parameter
        """
        Updates player's position and state based on emotion and platform collisions.
        Returns False if the player falls off the screen (game over).
        """
        current_time = time.time()
        
        # Manage grow up state
        if self.is_grown_up:
            if current_time - self.grow_up_start_time > self.grow_up_duration:
                self.is_grown_up = False
                self.width = self.original_width
                self.height = self.original_height
                print("Player shrunk back to normal.")
        
        # If emotion detection is active and no face is detected, stop horizontal movement
        if not face_detected_status:
            self.vel_x = 0
        else:
            # Handle emotion-based horizontal movement and skill activation
            if emotion == 'neutral':
                self.vel_x = 0
            elif emotion == 'happy':
                self.vel_x = PLAYER_SPEED
            elif emotion == 'surprise':
                self.vel_x = PLAYER_SPEED
                # Allow jump only if emotion changes to surprise and jumps are available
                if emotion != self.last_emotion and self.jump_count < self.max_jumps:
                    self.vel_y = JUMP_STRENGTH
                    self.jump_count += 1
                    self.on_ground = False # Player is no longer on ground after jumping
                    if game_sounds: # Play jump sound
                        game_sounds.play_jump_sound()
            elif emotion == 'sad':
                self.vel_x = -PLAYER_SPEED
            elif emotion == 'angry':
                # Activate grow up skill if not already grown up and cooldown is ready
                if not self.is_grown_up and (current_time - self.last_grow_up_time) >= self.grow_up_cooldown:
                    self.is_grown_up = True
                    self.grow_up_start_time = current_time
                    self.last_grow_up_time = current_time # Reset cooldown from activation
                    self.width = int(self.original_width * 1.5) # Increase size
                    self.height = int(self.original_height * 1.5) # Increase size
                    print("Player grew up! Can now stomp enemies.")
                    if game_sounds: # Play power-up sound
                        game_sounds.play_power_up_sound()
        
        self.last_emotion = emotion # Store current emotion for next frame's comparison
        
        # Apply gravity to vertical velocity
        self.vel_y += GRAVITY
        self.y += self.vel_y # Update vertical position
        
        # Keep player horizontally centered (game world scrolls around player)
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        
        # Platform collision (vertical only)
        self.on_ground = False
        for platform in platforms:
            # Check if player is colliding with the top of a platform and moving downwards
            if (self.x + self.width > platform.x and 
                self.x < platform.x + platform.width and
                self.y + self.height > platform.y and 
                self.y + self.height < platform.y + platform.height + 20 and # A small buffer for collision
                self.vel_y > 0): # Only if falling
                
                self.y = platform.y - self.height # Snap player to top of platform
                self.vel_y = 0 # Stop vertical movement
                self.on_ground = True # Player is on ground
                self.jump_count = 0 # Reset jump count
        
        # Check if player has fallen off the bottom of the screen
        if self.y > SCREEN_HEIGHT:
            return False # Game over
        
        return True # Player is still in game
    
    def draw(self, screen):
        """Draws the Mario-like player character on the screen."""
        current_width = self.width
        current_height = self.height
        
        # Adjust colors if grown up
        body_color = MARIO_BLUE if not self.is_grown_up else (100, 100, 255) # Lighter blue for grown up
        shirt_color = MARIO_RED if not self.is_grown_up else (255, 100, 100) # Lighter red for grown up

        # Body (Overalls - Blue)
        pygame.draw.rect(screen, body_color, (self.x + current_width * 0.15, self.y + current_height * 0.375, current_width * 0.7, current_height * 0.625), border_radius=int(current_width * 0.1))
        # Shirt (Red)
        pygame.draw.rect(screen, shirt_color, (self.x, self.y + current_height * 0.25, current_width, current_height * 0.5), border_radius=int(current_width * 0.1))
        # Head (Skin color)
        pygame.draw.circle(screen, MARIO_SKIN, (int(self.x + current_width//2), int(self.y + current_height * 0.2)), int(current_width * 0.25))
        # Hat (Red)
        pygame.draw.rect(screen, shirt_color, (self.x + current_width * 0.05, self.y, current_width * 0.9, current_height * 0.25), border_radius=int(current_width * 0.1))
        pygame.draw.rect(screen, shirt_color, (self.x - current_width * 0.15, self.y + current_height * 0.125, current_width * 0.35, current_height * 0.125), border_radius=int(current_width * 0.05)) # Hat brim
        # Eyes (Black)
        pygame.draw.circle(screen, BLACK, (int(self.x + current_width * 0.3), int(self.y + current_height * 0.175)), int(current_width * 0.075))
        pygame.draw.circle(screen, BLACK, (int(self.x + current_width * 0.7), int(self.y + current_height * 0.175)), int(current_width * 0.075))
        # Mustache (Brown)
        pygame.draw.line(screen, MARIO_GROUND_BROWN, (self.x + current_width * 0.3, self.y + current_height * 0.3), (self.x + current_width * 0.7, self.y + current_height * 0.3), int(current_width * 0.075))
        # Shoes (Brown)
        pygame.draw.rect(screen, MARIO_GROUND_BROWN, (self.x + current_width * 0.05, self.y + current_height - current_height * 0.125, current_width * 0.9, current_height * 0.125), border_radius=int(current_width * 0.05))


class Platform:
    """
    Represents a platform in the game world.
    Draws a Mario-like ground block.
    """
    def __init__(self, x, y, width):
        self.x, self.y, self.width = x, y, width
        self.height = 20
        # Cached Surface creation for performance: draw the platform once
        # and then blit it, rather than redrawing all shapes every frame.
        total_height = self.height + 30 # Includes the brown block part
        self.cached_surf = pygame.Surface((self.width, total_height), pygame.SRCALPHA)

        # Draw platform on cached surface (Mario-like block)
        # Top ground (green)
        pygame.draw.rect(self.cached_surf, MARIO_GROUND_TOP_GREEN, (0, 0, self.width, self.height))
        # Bottom layer (brown)
        pygame.draw.rect(self.cached_surf, MARIO_GROUND_BROWN, (0, self.height, self.width, 30))
        # Add some texture/lines for block appearance
        for i in range(0, self.width, 10):
            pygame.draw.line(self.cached_surf, (150, 100, 50), (i, self.height), (i, self.height + 30), 1)
        for i in range(0, 30, 10):
            pygame.draw.line(self.cached_surf, (150, 100, 50), (0, self.height + i), (self.width, self.height + i), 1)
    
    def update(self, effective_platform_speed=0):
        """Updates the platform's horizontal position based on world scrolling."""
        self.x += effective_platform_speed

    def draw(self, screen):
        """Draws the platform on the screen using its cached surface."""
        screen.blit(self.cached_surf, (self.x, self.y))

class Obstacle:
    """
    Base class for static obstacles (Goomba-like).
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        
    def update(self, effective_platform_speed=0):
        """Updates the obstacle's horizontal position based on world scrolling."""
        self.x += effective_platform_speed
        
    def draw(self, screen):
        """Draws a simple Goomba-like obstacle."""
        # Body (brown mushroom shape)
        pygame.draw.ellipse(screen, GOOMBA_BROWN, (self.x, self.y, self.width, self.height))
        # Feet (darker brown)
        pygame.draw.rect(screen, GOOMBA_FEET, (self.x + 5, self.y + self.height - 5, self.width - 10, 5))
        # Eyes (white with black pupils)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width * 0.3), int(self.y + self.height * 0.3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.3), int(self.y + self.height * 0.3)), 1)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width * 0.7), int(self.y + self.height * 0.3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.7), int(self.y + self.height * 0.3)), 1)

class MovingObstacle(Obstacle):
    """
    Represents a moving obstacle (Koopa Troopa-like).
    Patrols a defined horizontal range.
    """
    def __init__(self, x, y, patrol_range=100, move_speed=2):
        super().__init__(x, y)
        self.start_x = x
        self.end_x = x + patrol_range
        self.move_speed = move_speed
        self.direction = 1 # 1 for right, -1 for left
        
    def update(self, effective_platform_speed=0):
        """
        Updates the moving obstacle's position, applying both world scrolling
        and its own patrol movement.
        """
        # Apply world scrolling (from base class)
        super().update(effective_platform_speed) 
        
        # Apply local movement
        self.x += self.move_speed * self.direction
        
        # Reverse direction if it hits patrol limits
        if self.direction == 1 and self.x >= self.end_x:
            self.direction = -1
        elif self.direction == -1 and self.x <= self.start_x:
            self.direction = 1
        
        # Adjust start_x and end_x to keep relative patrol range with world movement.
        # This ensures the patrol range moves with the scrolling world.
        self.start_x += effective_platform_speed
        self.end_x += effective_platform_speed

    def draw(self, screen):
        """Draws a Koopa Troopa-like obstacle."""
        # Shell (green)
        pygame.draw.ellipse(screen, KOOPA_GREEN, (self.x, self.y, self.width, self.height))
        # Shell outline/pattern
        pygame.draw.ellipse(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.3), int(self.y + self.height * 0.3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.7), int(self.y + self.height * 0.3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.5), int(self.y + self.height * 0.6)), 3)

        # Head (skin color)
        pygame.draw.circle(screen, KOOPA_SKIN, (int(self.x + self.width * 0.5), int(self.y + self.height * 0.2)), 8)
        # Eyes
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width * 0.4), int(self.y + self.height * 0.15)), 2)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.4), int(self.y + self.height * 0.15)), 1)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width * 0.6), int(self.y + self.height * 0.15)), 2)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.6), int(self.y + self.height * 0.15)), 1)
        
        # Feet (brown)
        pygame.draw.rect(screen, MARIO_GROUND_BROWN, (self.x + 5, self.y + self.height - 5, 8, 5))
        pygame.draw.rect(screen, MARIO_GROUND_BROWN, (self.x + self.width - 13, self.y + self.height - 5, 8, 5))


class Cloud:
    """
    Represents a background cloud that moves independently.
    """
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.speed = random.uniform(0.5, 1.5) # Random speed for varied cloud movement
        
    def update(self):
        """Updates the cloud's position, wrapping around the screen."""
        self.x -= self.speed * 0.5 # Slower cloud movement
        if self.x < -self.size: # If cloud moves off screen to the left
            self.x = SCREEN_WIDTH + random.randint(50, 200) # Reset to right side
        
    def draw(self, screen):
        """Draws a fluffy cloud shape."""
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.size//2)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.size//3), int(self.y)), self.size//3)
        pygame.draw.circle(screen, WHITE, (int(self.x - self.size//3), int(self.y)), self.size//3)

class Collectible:
    """
    Represents a collectible item (star/coin).
    Bounces vertically and scrolls with the world.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 20
        self.height = 20
        self.bounce = 0 # Used for vertical bouncing animation
        
    def update(self, effective_platform_speed=0):
        """Updates the collectible's position and bounce animation."""
        self.x += effective_platform_speed # Apply world scrolling
        self.bounce += 0.1 # Increment bounce phase for animation
        
    def draw(self, screen):
        """Draws a spinning star collectible with a bouncing effect."""
        bounce_y = self.y + math.sin(self.bounce) * 5 # Calculate vertical offset for bounce
        
        # Draw a star shape using polygon points
        points = []
        for i in range(5):
            angle = math.pi/2 + i * (2 * math.pi / 5)
            x_outer = self.x + self.width/2 + self.width/2 * math.cos(angle)
            y_outer = bounce_y + self.height/2 - self.height/2 * math.sin(angle)
            points.append((x_outer, y_outer))
            
            angle_inner = angle + math.pi / 5
            x_inner = self.x + self.width/2 + self.width/4 * math.cos(angle_inner)
            y_inner = bounce_y + self.height/2 - self.height/4 * math.sin(angle_inner)
            points.append((x_inner, y_inner))
            
        pygame.draw.polygon(screen, MARIO_YELLOW, points)
        pygame.draw.polygon(screen, BLACK, points, 1) # Outline

def show_loading_screen(screen):
    """Displays a simple loading screen."""
    screen.fill((0, 0, 0)) # Black background
    font = pygame.font.Font(None, 48)
    text = font.render("Loading…", True, (255, 255, 255)) # White text
    # Center the text on the screen
    rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
    screen.blit(text, rect)
    pygame.display.flip() # Update the display to show the loading screen

class GameSounds:
    """
    Manages loading and playing game sound effects and music.
    Applies master, music, and sound effect volume settings.
    """
    def __init__(self, settings): # Pass settings object to GameSounds
        self.settings = settings
        self.sounds_dir = "sounds" # Directory where sound files are stored
        self.music_loaded = False # Track if music is loaded
        self.jump_sound = None
        self.coin_sound = None
        self.game_over_sound = None
        self.stomp_sound = None
        self.power_up_sound = None
        self.load_sounds()
        self.apply_volumes() # Apply initial volumes after loading

    def load_sounds(self):
        """Loads all sound effects and background music."""
        try:
            # Load background music
            music_path = os.path.join(self.sounds_dir, "game_music.mp3") # Or .mp3
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                self.music_loaded = True
                print(f"Loaded music: {music_path}")
            else:
                print(f"Music file not found: {music_path}")
                self.music_loaded = False

            # Load sound effects
            self.jump_sound = self._load_effect("jump.mp3")
            self.coin_sound = self._load_effect("coin.mp3")
            self.game_over_sound = self._load_effect("game_over.mp3")
            self.stomp_sound = self._load_effect("stomp.mp3")
            self.power_up_sound = self._load_effect("power_up.mp3")

        except pygame.error as e:
            print(f"Error loading sound: {e}")
            print("Sound will be disabled.")
            self.music_loaded = False
            self.jump_sound = None
            self.coin_sound = None
            self.game_over_sound = None
            self.stomp_sound = None
            self.power_up_sound = None

    def _load_effect(self, filename):
        """Helper to load a single sound effect."""
        filepath = os.path.join(self.sounds_dir, filename)
        if os.path.exists(filepath):
            return pygame.mixer.Sound(filepath)
        else:
            print(f"Sound effect file not found: {filepath}")
            return None

    def apply_volumes(self):
        """Applies the current volume settings to music and all loaded sound effects."""
        overall_music_volume = self.settings.master_volume * self.settings.music_volume
        pygame.mixer.music.set_volume(overall_music_volume)

        overall_sfx_volume = self.settings.master_volume * self.settings.sfx_volume
        if self.jump_sound: self.jump_sound.set_volume(overall_sfx_volume)
        if self.coin_sound: self.coin_sound.set_volume(overall_sfx_volume)
        if self.game_over_sound: self.game_over_sound.set_volume(overall_sfx_volume)
        if self.stomp_sound: self.stomp_sound.set_volume(overall_sfx_volume)
        if self.power_up_sound: self.power_up_sound.set_volume(overall_sfx_volume)

    def play_music(self):
        """Plays the background music, looping indefinitely."""
        if self.music_loaded:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        else:
            print("No music loaded, skipping play_music().")
    
    def stop_music(self):
        """Stops the background music."""
        pygame.mixer.music.stop()

    def play_jump_sound(self):
        """Plays the jump sound effect."""
        if self.jump_sound:
            self.jump_sound.play()

    def play_coin_sound(self):
        """Plays the coin collection sound effect."""
        if self.coin_sound:
            self.coin_sound.play()
            
    def play_game_over_sound(self):
        """Plays the game over sound effect."""
        if self.game_over_sound:
            self.game_over_sound.play()
            
    def play_stomp_sound(self):
        """Plays the enemy stomp sound effect."""
        if self.stomp_sound:
            self.stomp_sound.play()
            
    def play_power_up_sound(self):
        """Plays the power-up (grow up) sound effect."""
        if self.power_up_sound:
            self.power_up_sound.play()

class Game:
    """
    Main game class, managing game state, objects, menus, and the game loop.
    """
    def __init__(self):
        self.settings = Settings()
        # Apply initial screen size from settings
        self.screen = self.settings.apply_screen_size(self.settings.current_size)
        pygame.display.set_caption("Mario-like Emotion Platformer")
        self.clock = pygame.time.Clock()
        self.running = True # Main game loop control
        self.paused = False # Game pause state

        # --- UI Font Loading ---
        try:
            self.font_path = "pixel_font.ttf"
            if os.path.exists(self.font_path):
                self.title_font = pygame.font.Font(self.font_path, 60)
                self.header_font = pygame.font.Font(self.font_path, 48)
                self.menu_font = pygame.font.Font(self.font_path, 36)
                self.hud_font = pygame.font.Font(self.font_path, 28)
                self.small_font = pygame.font.Font(self.font_path, 22)
                self.button_font = pygame.font.Font(self.font_path, 30)
            else:
                print(f"Font file '{self.font_path}' not found. Using default font.")
                self.title_font = pygame.font.Font(None, 72)
                self.header_font = pygame.font.Font(None, 60)
                self.menu_font = pygame.font.Font(None, 48)
                self.hud_font = pygame.font.Font(None, 36)
                self.small_font = pygame.font.Font(None, 24)
                self.button_font = pygame.font.Font(None, 36)
        except Exception as e:
            print(f"Font loading error: {e}. Using default font.")
            self.title_font = pygame.font.Font(None, 72)
            self.header_font = pygame.font.Font(None, 60)
            self.menu_font = pygame.font.Font(None, 48)
            self.hud_font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            self.button_font = pygame.font.Font(None, 36)

        # Show loading screen before heavy initialization
        self.show_loading_screen(self.screen) # Changed to self.show_loading_screen

        # Initialize sound manager, passing the settings object
        self.game_sounds = GameSounds(self.settings)

        # Initialize game objects
        self.player = Player(100, 300)
        self.emotion_detector = EmotionDetector()

        # Game world objects
        self.platforms = [Platform(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH * 2)] # Initial long ground platform
        self.platforms.append(Platform(SCREEN_WIDTH * 0.8, SCREEN_HEIGHT - 150, 150)) #
        self.platforms.append(Platform(SCREEN_WIDTH * 1.2, SCREEN_HEIGHT - 250, 200)) #

        self.obstacles = [] #
        self.clouds = [Cloud(random.randint(0, SCREEN_WIDTH), random.randint(50, 200), random.randint(40, 80)) for _ in range(5)] #
        self.collectibles = [] #

        # Game state variables
        self.score = 0 #
        self.distance = 0 #
        self.game_over = False #
        # self.font and self.small_font are now instance variables from above

        # Start emotion detection in a separate thread if model and webcam are available
        if self.emotion_detector.model and self.emotion_detector.webcam: #
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop) #
            self.emotion_thread.daemon = True # Daemon thread exits when main program exits #
            self.emotion_thread.start() #

    # Make show_loading_screen a method of Game class to access fonts
    def show_loading_screen(self, screen): # Added self
        """Displays an improved loading screen."""
        screen_width = screen.get_width() # Use passed screen's width
        screen_height = screen.get_height() # Use passed screen's height

        screen.fill(BLACK) # Black background

        title_text_surf = self.title_font.render("Emotion Runner", True, MARIO_SKY_BLUE)
        title_rect = title_text_surf.get_rect(center=(screen_width // 2, screen_height // 2 - 50))
        screen.blit(title_text_surf, title_rect)

        loading_text_surf = self.menu_font.render("Loading…", True, WHITE) #
        loading_rect = loading_text_surf.get_rect(center=(screen_width // 2, screen_height // 2 + 20)) #
        screen.blit(loading_text_surf, loading_rect) #

        # Simple animated progress bar
        progress_bar_width = 200
        progress_bar_height = 20
        progress_bar_x = screen_width // 2 - progress_bar_width // 2
        progress_bar_y = screen_height // 2 + 70

        for i in range(progress_bar_width + 1):
            pygame.draw.rect(screen, MARIO_GROUND_TOP_GREEN, (progress_bar_x, progress_bar_y, i, progress_bar_height))
            pygame.draw.rect(screen, WHITE, (progress_bar_x, progress_bar_y, progress_bar_width, progress_bar_height), 2) # Border
            pygame.display.flip()
            pygame.time.delay(5) # Adjust for speed of loading bar

        pygame.display.flip() # Update the display to show the loading screen #

    def show_caution_screen(self):
        """
        Displays a caution message regarding glasses for emotion detection
        and waits for user input to continue.
        """
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        caution_font = pygame.font.Font(None, 60)
        instruction_font = pygame.font.Font(None, 40)
        
        caution_text = caution_font.render("CAUTION!", True, MARIO_RED)
        message_text1 = instruction_font.render("For optimal emotion detection,", True, BLACK)
        message_text2 = instruction_font.render("please play without glasses.", True, BLACK)
        continue_text = instruction_font.render("Press any key to continue...", True, BLACK)
        
        # Get rectangles for centering text
        caution_rect = caution_text.get_rect(center=(screen_width // 2, screen_height // 2 - 100))
        message1_rect = message_text1.get_rect(center=(screen_width // 2, screen_height // 2 - 30))
        message2_rect = message_text2.get_rect(center=(screen_width // 2, screen_height // 2 + 20))
        continue_rect = continue_text.get_rect(center=(screen_width // 2, screen_height // 2 + 100))
        
        self.screen.fill(MARIO_SKY_BLUE) # Fill background
        self.screen.blit(caution_text, caution_rect)
        self.screen.blit(message_text1, message1_rect)
        self.screen.blit(message_text2, message2_rect)
        self.screen.blit(continue_text, continue_rect)
        pygame.display.flip() # Update display
        
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    waiting_for_input = False # Exit loop on any key press

    def show_settings_menu(self):
        """
        Displays the settings menu, allowing the user to change screen size and sound volume.
        Uses keyboard input for selection.
        """
        button_width = 400 # Wider for longer text
        button_height = 60 #
        button_spacing = 15 #
        volume_bar_width = 200
        volume_bar_height = 20

        size_names = list(SCREEN_SIZES.keys()) #

        KEY_MASTER_VOL_UP = pygame.K_KP_PLUS # Numpad + #
        KEY_MASTER_VOL_DOWN = pygame.K_KP_MINUS # Numpad - #
        KEY_MUSIC_VOL_UP = pygame.K_UP # Up arrow #
        KEY_MUSIC_VOL_DOWN = pygame.K_DOWN # Down arrow #
        KEY_SFX_VOL_UP = pygame.K_RIGHT # Right arrow #
        KEY_SFX_VOL_DOWN = pygame.K_LEFT # Left arrow #

        while True:
            screen_width = self.screen.get_width() #
            screen_height = self.screen.get_height() #

            title_text_surf = self.header_font.render("Settings", True, BLACK) # Use header_font #
            title_rect = title_text_surf.get_rect(center=(screen_width//2, 80)) # Adjusted position #

            # Screen size buttons
            num_size_buttons = len(size_names)
            total_size_button_height = num_size_buttons * button_height + (num_size_buttons -1) * button_spacing
            start_y_screens = title_rect.bottom + 50

            size_buttons_with_keys = [] #
            for i, size_name in enumerate(size_names): #
                key = pygame.K_1 + i # Assign keys 1, 2, 3... #
                button_text = f"{i+1}. {size_name}" #
                if size_name != "Fullscreen": #
                    button_text += f" ({SCREEN_SIZES[size_name][0]}x{SCREEN_SIZES[size_name][1]})" #

                button = Button(
                    screen_width//2 - button_width//2, #
                    start_y_screens + i * (button_height + button_spacing), #
                    button_width, #
                    button_height, #
                    button_text,
                    self.button_font # Use button_font
                )
                size_buttons_with_keys.append((button, size_name, key)) #

            # Volume controls display (text and bars)
            volume_section_y_start = start_y_screens + total_size_button_height + 50
            text_y_offset = 0

            # Master Volume
            master_vol_text_surf = self.menu_font.render(f"Master: {int(self.settings.master_volume * 100)}% (+/- Numpad)", True, BLACK) #
            master_vol_rect = master_vol_text_surf.get_rect(midleft=(screen_width//2 - button_width//2, volume_section_y_start + text_y_offset)) #
            master_bar_x = master_vol_rect.right + 20
            master_filled_width = int(volume_bar_width * self.settings.master_volume)

            text_y_offset += 40
            # Music Volume
            music_vol_text_surf = self.menu_font.render(f"Music: {int(self.settings.music_volume * 100)}% (Up/Down)", True, BLACK) #
            music_vol_rect = music_vol_text_surf.get_rect(midleft=(screen_width//2 - button_width//2, volume_section_y_start + text_y_offset)) #
            music_bar_x = music_vol_rect.right + 20
            music_filled_width = int(volume_bar_width * self.settings.music_volume)

            text_y_offset += 40
            # SFX Volume
            sfx_vol_text_surf = self.menu_font.render(f"SFX: {int(self.settings.sfx_volume * 100)}% (Left/Right)", True, BLACK) #
            sfx_vol_rect = sfx_vol_text_surf.get_rect(midleft=(screen_width//2 - button_width//2, volume_section_y_start + text_y_offset)) #
            sfx_bar_x = sfx_vol_rect.right + 20
            sfx_filled_width = int(volume_bar_width * self.settings.sfx_volume)


            KEY_BACK = pygame.K_ESCAPE # Assign ESC key for back #
            back_button = Button(
                screen_width//2 - button_width//2, #
                volume_section_y_start + text_y_offset + 50, # Position below volume texts #
                button_width, #
                button_height, #
                "Back (ESC)", # Added key hint #
                self.button_font # Use button_font
            )

            self.screen.fill(MARIO_SKY_BLUE) #
            self.screen.blit(title_text_surf, title_rect) #

            for button, _, _ in size_buttons_with_keys: #
                button.draw(self.screen) #

            # Draw Master Volume
            self.screen.blit(master_vol_text_surf, master_vol_rect) #
            pygame.draw.rect(self.screen, WHITE, (master_bar_x, master_vol_rect.centery - volume_bar_height//2, volume_bar_width, volume_bar_height))
            pygame.draw.rect(self.screen, MARIO_GROUND_TOP_GREEN, (master_bar_x, master_vol_rect.centery - volume_bar_height//2, master_filled_width, volume_bar_height))
            pygame.draw.rect(self.screen, BLACK, (master_bar_x, master_vol_rect.centery - volume_bar_height//2, volume_bar_width, volume_bar_height), 2)


            # Draw Music Volume
            self.screen.blit(music_vol_text_surf, music_vol_rect) #
            pygame.draw.rect(self.screen, WHITE, (music_bar_x, music_vol_rect.centery - volume_bar_height//2, volume_bar_width, volume_bar_height))
            pygame.draw.rect(self.screen, MARIO_GROUND_TOP_GREEN, (music_bar_x, music_vol_rect.centery - volume_bar_height//2, music_filled_width, volume_bar_height))
            pygame.draw.rect(self.screen, BLACK, (music_bar_x, music_vol_rect.centery - volume_bar_height//2, volume_bar_width, volume_bar_height), 2)

            # Draw SFX Volume
            self.screen.blit(sfx_vol_text_surf, sfx_vol_rect) #
            pygame.draw.rect(self.screen, WHITE, (sfx_bar_x, sfx_vol_rect.centery - volume_bar_height//2, volume_bar_width, volume_bar_height))
            pygame.draw.rect(self.screen, MARIO_GROUND_TOP_GREEN, (sfx_bar_x, sfx_vol_rect.centery - volume_bar_height//2, sfx_filled_width, volume_bar_height))
            pygame.draw.rect(self.screen, BLACK, (sfx_bar_x, sfx_vol_rect.centery - volume_bar_height//2, volume_bar_width, volume_bar_height), 2)

            back_button.draw(self.screen) #
            pygame.display.flip() #

            for event in pygame.event.get(): #
                if event.type == pygame.QUIT: #
                    self.running = False #
                    pygame.quit() #
                    sys.exit() #

                # Handle button clicks via mouse
                if back_button.handle_event(event):
                    return
                for button, size_name, _ in size_buttons_with_keys:
                    if button.handle_event(event):
                        self.screen = self.settings.apply_screen_size(size_name) #
                        # Redraw with new dimensions in next loop
                        break

                if event.type == pygame.KEYDOWN: #
                    if event.key == KEY_BACK: #
                        return # Exit settings menu on ESC #

                    for button, size_name, key in size_buttons_with_keys: #
                        if event.key == key: #
                            self.screen = self.settings.apply_screen_size(size_name) #
                            break # Exit inner loop once a button is handled #

                    volume_change_amount = 0.05 # 5% increment/decrement #

                    if event.key == KEY_MASTER_VOL_UP: #
                        self.settings.set_master_volume(self.settings.master_volume + volume_change_amount) #
                        self.game_sounds.apply_volumes() #
                    elif event.key == KEY_MASTER_VOL_DOWN: #
                        self.settings.set_master_volume(self.settings.master_volume - volume_change_amount) #
                        self.game_sounds.apply_volumes() #
                    elif event.key == KEY_MUSIC_VOL_UP: #
                        self.settings.set_music_volume(self.settings.music_volume + volume_change_amount) #
                        self.game_sounds.apply_volumes() #
                    elif event.key == KEY_MUSIC_VOL_DOWN: #
                        self.settings.set_music_volume(self.settings.music_volume - volume_change_amount) #
                        self.game_sounds.apply_volumes() #
                    elif event.key == KEY_SFX_VOL_UP: #
                        self.settings.set_sfx_volume(self.settings.sfx_volume + volume_change_amount) #
                        self.game_sounds.apply_volumes() #
                    elif event.key == KEY_SFX_VOL_DOWN: #
                        self.settings.set_sfx_volume(self.settings.sfx_volume - volume_change_amount) #
                        self.game_sounds.apply_volumes() #

    def show_main_menu(self):
        """
        Displays the main menu with options to start the game, go to settings, or quit.
        Uses keyboard input for selection.
        """
        self.game_sounds.play_music() # Start playing menu music #

        button_width = 350 # Slightly wider for new font
        button_height = 70 # Slightly taller
        button_spacing = 25

        KEY_START = pygame.K_1 #
        KEY_SETTINGS = pygame.K_2 #
        KEY_QUIT = pygame.K_3 #

        while True:
            screen_width = self.screen.get_width() #
            screen_height = self.screen.get_height() #

            start_y = screen_height // 2 - (button_height * 1.5 + button_spacing) # Adjust centering

            start_button = Button(
                screen_width//2 - button_width//2, #
                start_y,
                button_width, #
                button_height, #
                f"1. Start Game", #
                self.button_font # Use the new button_font
            )
            settings_button = Button(
                screen_width//2 - button_width//2, #
                start_y + button_height + button_spacing, #
                button_width, #
                button_height, #
                f"2. Settings", #
                self.button_font # Use the new button_font
            )
            quit_button = Button(
                screen_width//2 - button_width//2, #
                start_y + (button_height + button_spacing) * 2, #
                button_width, #
                button_height, #
                f"3. Quit Game", #
                self.button_font # Use the new button_font
            )

            title_text = self.title_font.render("Emotion Platformer", True, BLACK) # # Use title_font
            title_rect = title_text.get_rect(center=(screen_width//2, screen_height // 4)) # Adjusted title position

            self.screen.fill(MARIO_SKY_BLUE) #
            self.screen.blit(title_text, title_rect) #
            start_button.draw(self.screen) #
            settings_button.draw(self.screen) #
            quit_button.draw(self.screen) #
            pygame.display.flip() # Update display #

            for event in pygame.event.get(): #
                if event.type == pygame.QUIT: #
                    self.running = False #
                    pygame.quit() #
                    sys.exit() #

                # Handle button clicks via mouse
                if start_button.handle_event(event):
                    self.game_sounds.stop_music() #
                    return
                if settings_button.handle_event(event):
                    self.show_settings_menu() #
                    self.game_sounds.apply_volumes() #
                    self.game_sounds.play_music() #
                if quit_button.handle_event(event):
                    self.running = False #
                    pygame.quit() #
                    sys.exit() #

                if event.type == pygame.KEYDOWN: #
                    if event.key == KEY_START: #
                        self.game_sounds.stop_music() # Stop menu music when starting game #
                        return  # Start the game, exit menu loop #
                    elif event.key == KEY_SETTINGS: #
                        self.show_settings_menu()  # Show settings menu, then return here #
                        self.game_sounds.apply_volumes() # Apply any changed volumes immediately #
                        self.game_sounds.play_music() # Ensure music continues after returning from settings #
                    elif event.key == KEY_QUIT: #
                        self.running = False #
                        pygame.quit() #
                        sys.exit() #
                    elif event.key == pygame.K_ESCAPE: # Allow ESC to quit from main menu #
                        self.running = False #
                        pygame.quit() #
                        sys.exit() #
        
    def emotion_detection_loop(self):
        """
        Runs in a separate thread to continuously detect emotion
        while the game is running.
        """
        while self.running:
            if self.emotion_detector.model and self.emotion_detector.webcam:
                self.emotion_detector.detect_emotion()
                # This waitKey is crucial for OpenCV to process frames from camera
                # and also for any OpenCV windows if they were used for debugging.
                if cv2.waitKey(1) & 0xFF == ord('q'): # Allows closing debug windows with 'q'
                    # This is mostly for debugging if you re-enable cv2.imshow somewhere
                    pass 
            else:
                # If webcam/model isn't active, prevent fast spinning
                time.sleep(0.1) 
    
    def handle_keyboard_controls(self):
        """
        Provides fallback keyboard controls for player movement
        if emotion detection is not active.
        """
        keys = pygame.key.get_pressed()
        
        # Check if the desired emotion for 'surprise' or 'angry' is *newly* pressed
        # This prevents continuous jump/power-up activation from holding the key
        
        # Store previous state for comparison
        prev_emotion = self.emotion_detector.current_emotion
        
        if keys[pygame.K_SPACE]:
            self.emotion_detector.current_emotion = 'surprise' # Maps to jump
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.emotion_detector.current_emotion = 'happy' # Maps to move forward
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.emotion_detector.current_emotion = 'sad' # Maps to move backward
        elif keys[pygame.K_RETURN]: # Use Enter for 'angry' (grow up)
            self.emotion_detector.current_emotion = 'angry'
        else:
            self.emotion_detector.current_emotion = 'neutral' # Maps to stop

        # If using keyboard, assume a 'face' is always detected for movement purposes.
        self.emotion_detector.face_detected = True 
    
    def spawn_platform(self):
        """
        Spawns new platforms dynamically as the player progresses.
        Ensures a continuous stream of platforms.
        """
        # Only spawn a new platform if the rightmost platform is almost visible
        # or if there are no platforms yet (initial spawn).
        if len(self.platforms) == 0 or self.platforms[-1].x < SCREEN_WIDTH - 100:
            x = SCREEN_WIDTH + random.randint(0, 100) # Randomize starting x slightly off-screen
            
            # Decide if it's a ground extension or a floating platform
            if random.random() < 0.6: # 60% chance for ground extension
                y = SCREEN_HEIGHT - 50 # Consistent ground level
                width = random.randint(150, 300) # Longer ground segments
            else: # 40% chance for floating platform
                y = random.randint(SCREEN_HEIGHT - 250, SCREEN_HEIGHT - 100) # Floating platforms above ground
                width = random.randint(80, 200) # Shorter floating platforms
            
            self.platforms.append(Platform(x, y, width))
    
    def spawn_obstacle(self):
        """
        Spawns obstacles (Goombas or Koopa Troopas) on existing platforms.
        """
        for platform in self.platforms:
            # Only spawn on platforms that are currently visible or just coming into view
            if (platform.x > SCREEN_WIDTH - 200 and platform.x < SCREEN_WIDTH - 100):
                if random.random() < 0.005: # Low chance to spawn static obstacles
                    self.obstacles.append(Obstacle(platform.x + random.randint(20, platform.width - 50), 
                                                 platform.y - 30)) # Position above platform
                elif random.random() < 0.007: # Slightly higher chance for moving obstacles
                    # Ensure patrol range fits within the platform or a reasonable area
                    patrol_start_x = platform.x + 20
                    patrol_end_x = platform.x + platform.width - 50
                    if patrol_end_x - patrol_start_x > 50: # Ensure enough space for patrol
                        self.obstacles.append(MovingObstacle(
                            patrol_start_x,
                            platform.y - 30,
                            patrol_range=random.randint(50, min(100, int(platform.width * 0.5))),
                            move_speed=random.uniform(1, 2.5)
                        ))
    
    def spawn_collectible(self):
        """
        Spawns collectibles (stars/coins) on existing platforms.
        """
        for platform in self.platforms:
            # Only spawn on platforms that are currently visible or just coming into view
            if (platform.x > SCREEN_WIDTH - 250 and platform.x < SCREEN_WIDTH - 150 and
                random.random() < 0.05): # Higher chance for collectibles
                self.collectibles.append(Collectible(platform.x + random.randint(10, platform.width - 40), 
                                                   platform.y - 40)) # Position above platform
    
    def check_collisions(self):
        """
        Checks for collisions between the player and obstacles/collectibles.
        Handles stomping non-moving enemies, grown-up player defeating enemies,
        and collecting items. Returns False if a fatal collision occurs.
        """
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        
        # Check obstacle collisions
        # Iterate over a copy of the list to safely remove items during iteration
        for obstacle in self.obstacles[:]: 
            obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)
            
            if player_rect.colliderect(obstacle_rect):
                # Stomp Condition for Non-Moving Obstacles:
                # 1. Player is falling (self.player.vel_y > 0).
                # 2. The obstacle is a non-moving type (Obstacle, not MovingObstacle).
                # 3. Player's feet are colliding with the top portion of the obstacle.
                
                # Player's feet must land within the top X% of the enemy's height.
                # e.g., if enemy height is 30, top 40% is 12 pixels.
                # This means player_rect.bottom should be between obstacle_rect.top and obstacle_rect.top + 12.
                stomp_effective_height = obstacle_rect.height * 0.4 

                is_stomp = (isinstance(obstacle, Obstacle) and
                            not isinstance(obstacle, MovingObstacle) and # Ensures it's the base Obstacle, not a MovingObstacle
                            self.player.vel_y > 0 and
                            player_rect.bottom > obstacle_rect.top and  # Player's feet are below the very top of obstacle (i.e. intersecting)
                            player_rect.bottom < obstacle_rect.top + stomp_effective_height) # Player's feet are within the 'stompable' top area

                if is_stomp:
                    self.obstacles.remove(obstacle)
                    self.score += 25 # Score for stomping a non-moving enemy
                    if self.game_sounds:
                        self.game_sounds.play_stomp_sound()
                    
                    # Give player a small bounce
                    self.player.vel_y = -7  # Negative value for upward movement; JUMP_STRENGTH is -15 for comparison
                    self.player.on_ground = False # Player is now airborne from the bounce
                    # Note: This bounce does not reset or consume the player's regular jump counts (e.g., double jump)
                    print("Non-moving enemy stomped!")
                
                elif self.player.is_grown_up: # If not a stomp, but player is grown up (this can defeat any obstacle type)
                    self.obstacles.remove(obstacle)
                    self.score += 50 # Score for defeating an enemy while grown up
                    if self.game_sounds:
                        self.game_sounds.play_stomp_sound() # Or a different "power defeat" sound could be used
                    print("Enemy defeated by grown-up player!")
                
                else: # Collision without a successful stomp or grow-up power (fatal for any obstacle type)
                    if self.game_sounds:
                        self.game_sounds.play_game_over_sound()
                    return False # Game over
        
        # Check collectible collisions (this part remains the same)
        for collectible in self.collectibles[:]: 
            collectible_rect = pygame.Rect(collectible.x, collectible.y, collectible.width, collectible.height)
            if player_rect.colliderect(collectible_rect):
                self.collectibles.remove(collectible)
                self.score += 10
                if self.game_sounds:
                    self.game_sounds.play_coin_sound()
        
        return True # No fatal collisions, or all collisions were handled (stomp, grow-up power, collectible)
    
    def update(self):
        """
        Updates the game state for one frame: player, platforms, obstacles, collectibles,
        and manages spawning and collisions.
        """
        if self.game_over:
            return # Do nothing if game is over
        
        # Handle keyboard as backup if emotion detection is not active
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            self.handle_keyboard_controls()
        
        current_emotion = self.emotion_detector.current_emotion
        face_detected_status = self.emotion_detector.face_detected # Get face detection status
        
        # Player update (vertical movement handled by gravity, horizontal by world scroll)
        if not self.player.update(current_emotion, self.platforms, face_detected_status, self.game_sounds): # Pass game_sounds
            self.game_over = True # Player fell off screen
            self.game_sounds.play_game_over_sound() # Play game over sound
            return
        
        # Check for collisions with obstacles and collectibles
        if not self.check_collisions():
            self.game_over = True # Player hit an obstacle (sound already played in check_collisions)
            return
        
        # Determine effective speed for world scrolling based on player's horizontal velocity
        effective_platform_speed = -self.player.vel_x
        
        # Update and clean up platforms
        for platform in self.platforms[:]: # Iterate over a copy
            platform.update(effective_platform_speed)
            # Remove platforms that have moved completely off-screen
            if platform.x + platform.width < 0 and effective_platform_speed < 0:
                self.platforms.remove(platform)
            # This condition might not be strictly needed if only moving left, but good for completeness
            elif platform.x > SCREEN_WIDTH and effective_platform_speed > 0:
                self.platforms.remove(platform)
        
        # Update and clean up obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update(effective_platform_speed)
            if obstacle.x + obstacle.width < 0 and effective_platform_speed < 0:
                self.obstacles.remove(obstacle)
            elif obstacle.x > SCREEN_WIDTH and effective_platform_speed > 0:
                self.obstacles.remove(obstacle)
        
        # Update and clean up collectibles
        for collectible in self.collectibles[:]:
            collectible.update(effective_platform_speed)
            if collectible.x + collectible.width < 0 and effective_platform_speed < 0:
                self.collectibles.remove(collectible)
            elif collectible.x > SCREEN_WIDTH and effective_platform_speed > 0:
                self.collectibles.remove(collectible)
        
        # Update clouds (they move independently of player's horizontal speed)
        for cloud in self.clouds:
            cloud.update()
        
        # Spawning logic: spawn new elements when player moves forward (and a face is detected or using keyboard)
        if (current_emotion == 'happy' or current_emotion == 'surprise') and face_detected_status:
            self.spawn_platform()
            self.spawn_obstacle()
            self.spawn_collectible()
            self.distance += abs(self.player.vel_x) # Increase distance based on player's forward movement
            if self.distance % 100 == 0: # Award score for distance milestones
                self.score += 1
        elif current_emotion == 'sad' and face_detected_status:
            pass # No new spawns when moving backward
        # If no face is detected, no new spawns and no distance increase.
    
    def draw_background(self):
        """Fills the screen with the sky blue background color."""
        self.screen.fill(MARIO_SKY_BLUE)
    
    def show_pause_menu(self):
        """Displays the pause menu overlay."""
        current_screen_width = self.screen.get_width()
        current_screen_height = self.screen.get_height()

        overlay = pygame.Surface((current_screen_width, current_screen_height)) #
        overlay.set_alpha(180) # 128 out of 255 for transparency #
        overlay.fill(BLACK) #
        self.screen.blit(overlay, (0, 0)) #

        pause_text_surf = self.header_font.render("PAUSED", True, WHITE) #
        continue_text_surf = self.menu_font.render("Press P to Continue", True, WHITE) #
        quit_text_surf = self.menu_font.render("Press ESC for Main Menu", True, WHITE) # Changed to Main Menu

        self.screen.blit(pause_text_surf, (current_screen_width//2 - pause_text_surf.get_width()//2, current_screen_height//2 - 100)) #
        self.screen.blit(continue_text_surf, (current_screen_width//2 - continue_text_surf.get_width()//2, current_screen_height//2)) #
        self.screen.blit(quit_text_surf, (current_screen_width//2 - quit_text_surf.get_width()//2, current_screen_height//2 + 60)) # Adjusted spacing

        pygame.display.flip() # Update the display to show the pause menu #

    def draw(self):
        """
        Draws all game elements on the screen: background, clouds, platforms,
        obstacles, collectibles, player, and UI elements.
        Also handles game over and pause screen overlays.
        """
        current_screen_width = self.screen.get_width() # Get current screen width
        current_screen_height = self.screen.get_height() # Get current screen height

        self.draw_background() #

        for cloud in self.clouds: #
            cloud.draw(self.screen) #

        for platform in self.platforms: #
            platform.draw(self.screen) #

        for obstacle in self.obstacles: #
            obstacle.draw(self.screen) #

        for collectible in self.collectibles: #
            collectible.draw(self.screen) #

        self.player.draw(self.screen) #

        # --- HUD Elements ---
        # Emotion Text
        emotion_text_surf = self.hud_font.render(f"Emotion: {self.emotion_detector.current_emotion}", True, BLACK) #
        self.screen.blit(emotion_text_surf, (10, 10)) #

        # Score Text
        score_text_surf = self.hud_font.render(f"Score: {self.score}", True, BLACK) #
        self.screen.blit(score_text_surf, (10, 10 + self.hud_font.get_height())) #

        # Distance Text
        distance_text_surf = self.hud_font.render(f"Distance: {self.distance//10}m", True, BLACK) #
        self.screen.blit(distance_text_surf, (10, 10 + self.hud_font.get_height() * 2)) #

        # FPS Counter
        fps = int(self.clock.get_fps()) #
        fps_text_surf = self.small_font.render(f"FPS: {fps}", True, BLACK) #
        # Position FPS counter next to camera feed if camera feed is on top right, otherwise top right.
        camera_feed_width_check = 200 # Expected width of camera feed
        fps_x_pos = current_screen_width - fps_text_surf.get_width() - 10
        if self.emotion_detector.model and self.emotion_detector.webcam:
             fps_x_pos = current_screen_width - camera_feed_width_check - fps_text_surf.get_width() - 20 # Adjust if cam feed is there

        self.screen.blit(fps_text_surf, (fps_x_pos, 10)) #


        # Grow-Up Status and Cooldown Bar
        bar_width = 200
        bar_height = 15
        bar_x = current_screen_width // 2 - bar_width // 2
        bar_y = 10

        if self.player.is_grown_up: #
            remaining_time = max(0, self.player.grow_up_duration - (time.time() - self.player.grow_up_start_time)) #
            status_text_surf = self.small_font.render(f"GROW UP!", True, MARIO_RED) #
            self.screen.blit(status_text_surf, (bar_x + bar_width // 2 - status_text_surf.get_width() // 2, bar_y + bar_height)) #

            fill_ratio = remaining_time / self.player.grow_up_duration
            pygame.draw.rect(self.screen, (200,200,200), (bar_x, bar_y, bar_width, bar_height)) # Background
            pygame.draw.rect(self.screen, MARIO_RED, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height)) # Foreground
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 2) # Border
        else:
            cooldown_remaining = max(0, self.player.grow_up_cooldown - (time.time() - self.player.last_grow_up_time)) #
            status_text_surf = self.small_font.render(f"Grow Up Ready!", True, MARIO_GROUND_TOP_GREEN) #
            if cooldown_remaining > 0: #
                status_text_surf = self.small_font.render(f"Cooldown", True, BLACK) #

            self.screen.blit(status_text_surf, (bar_x + bar_width // 2 - status_text_surf.get_width() // 2, bar_y + bar_height))

            fill_ratio = 1.0 - (cooldown_remaining / self.player.grow_up_cooldown) if self.player.grow_up_cooldown > 0 else 1.0
            bar_color = MARIO_GROUND_TOP_GREEN if cooldown_remaining == 0 else BUTTON_HOVER

            pygame.draw.rect(self.screen, (200,200,200), (bar_x, bar_y, bar_width, bar_height)) # Background
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height)) # Foreground
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 2) # Border


        # Instructions
        instruction_y_start = current_screen_height - 30 #
        if not self.emotion_detector.model or not self.emotion_detector.webcam: #
            instruction_text_surf = self.small_font.render("Arrows/D/A: Move, SPACE: Jump, ENTER: Grow, P: Pause", True, BLACK) #
            self.screen.blit(instruction_text_surf, (10, instruction_y_start)) #
        else:
            instruction_text1_surf = self.small_font.render("HAPPY: ->, SURPRISE: Jump, SAD: <-, ANGRY: Grow, P: Pause", True, BLACK) #
            self.screen.blit(instruction_text1_surf, (10, instruction_y_start - self.small_font.get_height() - 2)) #

            if not self.emotion_detector.face_detected: #
                instruction_text2_surf = self.small_font.render("NO FACE DETECTED - Look at camera!", True, MARIO_RED) #
            else:
                instruction_text2_surf = self.small_font.render("Show emotions to control!", True, BLACK) #
            self.screen.blit(instruction_text2_surf, (10, instruction_y_start)) #


        # --- Draw Camera Feed ---
        if self.emotion_detector.model and self.emotion_detector.webcam:
            frame_to_display = None
            # Safely get the latest frame
            with self.emotion_detector.frame_lock:
                if self.emotion_detector.latest_display_frame is not None:
                    frame_to_display = self.emotion_detector.latest_display_frame.copy() # Make a copy to work on
            
            if frame_to_display is not None:
                try:
                    # OpenCV frame is BGR, Pygame needs RGB.
                    frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                    
                    # OpenCV frame is (height, width, channels). Pygame surfarray needs (width, height, channels).
                    frame_rgb_transposed = frame_rgb.transpose([1, 0, 2])
                    
                    # Create Pygame surface.
                    camera_surface = pygame.surfarray.make_surface(frame_rgb_transposed)
                    
                    # Optional: Flip if the camera image is mirrored (common for selfie view)
                    # camera_surface = pygame.transform.flip(camera_surface, True, False) 

                    # Position for the camera feed (e.g., top-right corner)
                    camera_feed_width = camera_surface.get_width()   # Should be 200
                    camera_feed_height = camera_surface.get_height() # Should be 150
                    
                    cam_x = current_screen_width - camera_feed_width - 10  # 10px padding
                    cam_y = 10  # 10px padding
                    
                    self.screen.blit(camera_surface, (cam_x, cam_y))
                    
                    # Draw a border around the camera feed
                    pygame.draw.rect(self.screen, BLACK, (cam_x - 2, cam_y - 2, camera_feed_width + 4, camera_feed_height + 4), 2)

                except Exception as e:
                    print(f"Error displaying camera feed: {e}")
                    # Optionally, draw a placeholder if processing fails
                    cam_x_err = current_screen_width - 200 - 10 
                    cam_y_err = 10
                    pygame.draw.rect(self.screen, (70,70,70), (cam_x_err, cam_y_err, 200, 150))
                    err_text_surf = self.small_font.render("Cam Error", True, WHITE)
                    self.screen.blit(err_text_surf, (cam_x_err + 10, cam_y_err + 10))


        if self.game_over: #
            overlay = pygame.Surface((current_screen_width, current_screen_height)) #
            overlay.set_alpha(180) # Semi-transparent overlay #
            overlay.fill(BLACK) #
            self.screen.blit(overlay, (0, 0)) #

            game_over_surf = self.title_font.render("GAME OVER", True, MARIO_RED) #
            final_score_surf = self.menu_font.render(f"Final Score: {self.score}", True, WHITE) #
            restart_surf = self.small_font.render("Press R to restart or ESC to quit", True, WHITE) #

            self.screen.blit(game_over_surf, (current_screen_width//2 - game_over_surf.get_width()//2, current_screen_height//2 - 80)) #
            self.screen.blit(final_score_surf, (current_screen_width//2 - final_score_surf.get_width()//2, current_screen_height//2 - 10)) #
            self.screen.blit(restart_surf, (current_screen_width//2 - restart_surf.get_width()//2, current_screen_height//2 + 40)) #

        if self.paused: #
            self.show_pause_menu() #

        pygame.display.flip() # Update the full display Surface to the screen #
    
    def restart_game(self):
        """Resets all game elements to their initial state for a new game."""
        self.player = Player(100, 300)
        # Reset initial platforms for a fresh start
        # Get current screen dimensions for platform reset
        current_screen_width_val = self.screen.get_width()
        current_screen_height_val = self.screen.get_height()

        self.platforms = [Platform(0, current_screen_height_val - 50, current_screen_width_val * 2)]
        self.platforms.append(Platform(current_screen_width_val * 0.8, current_screen_height_val - 150, 150))
        self.platforms.append(Platform(current_screen_width_val * 1.2, current_screen_height_val - 250, 200))


        self.obstacles = []
        self.collectibles = []
        self.score = 0
        self.distance = 0
        self.game_over = False
        self.game_sounds.play_music() # Start playing game music again on restart
    
    def run(self):
        """
        The main game loop. Handles events, updates game state, and draws to the screen.
        """
        # Show caution screen first if emotion detection is active
        if self.emotion_detector.model and self.emotion_detector.webcam:
            self.show_caution_screen()

        # Show main menu before starting the game loop
        self.show_main_menu() 
        
        # Once game starts from main menu, play game music
        self.game_sounds.play_music()
        
        while self.running:
            # Event handling for the main game loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False # Set running to False to exit game loop
                elif event.type == pygame.KEYDOWN:
                    if self.game_over: # Only these keys active if game over
                        if event.key == pygame.K_ESCAPE:
                            self.running = False 
                        elif event.key == pygame.K_r:
                            self.restart_game()
                    elif self.paused: # Only these keys active if paused
                         if event.key == pygame.K_p:
                            self.paused = not self.paused 
                            if not self.paused:
                                pygame.mixer.music.unpause()
                         elif event.key == pygame.K_ESCAPE: # Back to main menu from pause
                            self.paused = False # Unpause first
                            pygame.mixer.music.unpause() # Ensure music is unpaused
                            self.game_sounds.stop_music() # Stop game music
                            self.show_main_menu() # Show main menu
                            self.game_sounds.play_music() # Start menu music (or game music if game is started again)
                    else: # Keys active during gameplay
                        if event.key == pygame.K_ESCAPE: # Back to main menu from gameplay (via pause)
                            self.paused = True
                            pygame.mixer.music.pause()
                            # The pause menu will handle going back to main menu
                        elif event.key == pygame.K_p: # Toggle pause
                            self.paused = not self.paused
                            if self.paused:
                                pygame.mixer.music.pause() 
                            else:
                                pygame.mixer.music.unpause()
            
            # Only update game logic if not paused and not game over
            if not self.paused and not self.game_over:
                self.update() 
            
            self.draw() 
            self.clock.tick(FPS) 
        
        # Cleanup: release webcam and destroy OpenCV windows when game loop exits
        if hasattr(self.emotion_detector, 'webcam') and self.emotion_detector.webcam:
            self.emotion_detector.webcam.release()
        cv2.destroyAllWindows()
        pygame.quit() 
        sys.exit() 

if __name__ == "__main__":
    game = Game()
    game.run()
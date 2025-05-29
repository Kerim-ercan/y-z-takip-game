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
    def __init__(self, x, y, width, height, text, font_size=48, text_color=WHITE, button_color=MARIO_GROUND_TOP_GREEN):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.text_color = text_color
        self.button_color = button_color
        self.is_hovered = False
        
    def draw(self, screen):
        """
        Draws the button on the screen, including its background, border, and text.
        Changes color on hover.
        """
        # Determine button color based on hover state
        color = BUTTON_HOVER if self.is_hovered else self.button_color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=10)  # Button border
        
        # Render and center the button text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        """
        Handles Pygame events for the button.
        Updates hover state on MOUSEMOTION.
        Returns True if the button is clicked (MOUSEBUTTONUP while hovered).
        """
        if event.type == pygame.MOUSEMOTION:
            # Update hover state when mouse moves over the button
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            # Check for click on mouse button release
            # This is more reliable than MOUSEBUTTONDOWN as it prevents accidental clicks
            # if the mouse is dragged off the button before release.
            if self.is_hovered:
                return True # Button was clicked
        return False # Button was not clicked or event not relevant

class Settings:
    """
    Manages game settings, including screen size.
    Loads and saves settings to a JSON file.
    """
    def __init__(self):
        self.settings_file = "game_settings.json"
        self.current_size = "Medium"  # Default size
        self.load_settings()
        
    def load_settings(self):
        """Loads screen size setting from file."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.current_size = settings.get('screen_size', 'Medium')
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Saves current screen size setting to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump({'screen_size': self.current_size}, f)
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
        and updates `self.current_emotion`. Displays the camera feed.
        """
        # If model or webcam not available, return the current emotion (likely 'neutral' or keyboard-set)
        if not self.model or not self.webcam:
            self.face_detected = False # No face detected if system not active
            return self.current_emotion
            
        try:
            ret, frame = self.webcam.read()
            if not ret: # If frame couldn't be read
                self.face_detected = False
                self.current_emotion = 'neutral' # Reset emotion if camera fails
                return self.current_emotion
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5) # Detect faces
            
            # Process only the first detected face for simplicity
            if len(faces) > 0:
                self.face_detected = True # Face detected!
                (x, y, w, h) = faces[0] # Get coordinates of the first face
                face_img = gray[y:y+h, x:x+w] # Extract face region
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw rectangle around face
                
                face_img = cv2.resize(face_img, (48, 48)) # Resize face image to 48x48 for model
                img_features = self.extract_features(face_img)
                prediction = self.model.predict(img_features, verbose=0) # Predict emotion
                emotion_label = self.labels[prediction.argmax()] # Get emotion label with highest probability
                
                cv2.putText(frame, emotion_label, (x-10, y-10), # Display emotion on frame
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                
                self.current_emotion = emotion_label # Update current emotion
            else:
                self.face_detected = False # No face detected
                self.current_emotion = 'neutral' # Stop character if no face is detected
            
            # Resize frame for displaying in a small OpenCV window
            small_frame = cv2.resize(frame, (200, 150))
            cv2.imshow("Emotion", small_frame) # Show the frame
            cv2.waitKey(1) # Wait for a short period to allow window updates
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            self.face_detected = False # Assume no face if error occurs
            self.current_emotion = 'neutral' # Stop character if error occurs
        
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
    """
    def __init__(self):
        self.sounds_dir = "sounds" # Directory where sound files are stored
        self.music = None
        self.music_loaded = False # Track if music is loaded
        self.jump_sound = None
        self.coin_sound = None
        self.game_over_sound = None
        self.stomp_sound = None
        self.power_up_sound = None
        self.load_sounds()
    
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
        
        # Show loading screen before heavy initialization
        show_loading_screen(self.screen)
        
        # Initialize sound manager
        self.game_sounds = GameSounds()

        # Initialize game objects
        self.player = Player(100, 300)
        self.emotion_detector = EmotionDetector()
        
        # Game world objects
        self.platforms = [Platform(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH * 2)] # Initial long ground platform
        self.platforms.append(Platform(SCREEN_WIDTH * 0.8, SCREEN_HEIGHT - 150, 150))
        self.platforms.append(Platform(SCREEN_WIDTH * 1.2, SCREEN_HEIGHT - 250, 200))
        
        self.obstacles = []
        self.clouds = [Cloud(random.randint(0, SCREEN_WIDTH), random.randint(50, 200), random.randint(40, 80)) for _ in range(5)]
        self.collectibles = []
        
        # Game state variables
        self.score = 0
        self.distance = 0
        self.game_over = False
        self.font = pygame.font.Font(None, 36) # Font for score, etc.
        self.small_font = pygame.font.Font(None, 24) # Smaller font for instructions
        
        # Start emotion detection in a separate thread if model and webcam are available
        if self.emotion_detector.model and self.emotion_detector.webcam:
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop)
            self.emotion_thread.daemon = True # Daemon thread exits when main program exits
            self.emotion_thread.start()

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
        Displays the settings menu, allowing the user to change screen size.
        Uses keyboard input for selection.
        """
        button_width = 300
        button_height = 60
        button_spacing = 20
        
        size_names = list(SCREEN_SIZES.keys())
        
        while True:
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            
            total_button_area_height = (len(SCREEN_SIZES) + 1) * (button_height + button_spacing)
            start_y = screen_height // 2 - (total_button_area_height // 2)
            
            size_buttons_with_keys = [] # Store button and its associated key
            for i, size_name in enumerate(size_names):
                key = pygame.K_1 + i # Assign keys 1, 2, 3...
                button_text = f"{i+1}. {size_name} ({SCREEN_SIZES[size_name][0]}x{SCREEN_SIZES[size_name][1]})" \
                              if size_name != "Fullscreen" else f"{i+1}. {size_name}"
                button = Button(
                    screen_width//2 - button_width//2,
                    start_y + i * (button_height + button_spacing),
                    button_width,
                    button_height,
                    button_text
                )
                size_buttons_with_keys.append((button, size_name, key))
            
            KEY_BACK = pygame.K_ESCAPE # Assign ESC key for back
            back_button = Button(
                screen_width//2 - button_width//2,
                start_y + len(SCREEN_SIZES) * (button_height + button_spacing),
                button_width,
                button_height,
                "Back (ESC)" # Added key hint
            )
            
            title_font = pygame.font.Font(None, 72)
            title_text = title_font.render("Settings", True, BLACK)
            title_rect = title_text.get_rect(center=(screen_width//2, 150))
            
            self.screen.fill(MARIO_SKY_BLUE)
            self.screen.blit(title_text, title_rect)
            for button, _, _ in size_buttons_with_keys:
                button.draw(self.screen)
            back_button.draw(self.screen)
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == KEY_BACK:
                        return # Exit settings menu on ESC
                    
                    for button, size_name, key in size_buttons_with_keys:
                        if event.key == key:
                            self.screen = self.settings.apply_screen_size(size_name)
                            # After changing screen size, the current loop iteration will
                            # redraw buttons with new dimensions.
                            break # Exit inner loop once a button is handled
                
                # Keep existing mouse hover functionality for visual feedback if desired
                for button, _, _ in size_buttons_with_keys:
                    button.handle_event(event)
                back_button.handle_event(event)
    def show_main_menu(self):
        """
        Displays the main menu with options to start the game, go to settings, or quit.
        Uses keyboard input for selection.
        """
        self.game_sounds.play_music() # Start playing menu music

        button_width = 300
        button_height = 60
        button_spacing = 20
        
        # Define key mappings for menu options
        KEY_START = pygame.K_1
        KEY_SETTINGS = pygame.K_2
        KEY_QUIT = pygame.K_3
        
        while True:
            # Recalculate screen dimensions in case they changed from settings menu
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            
            # Calculate starting Y position to center the buttons vertically
            start_y = screen_height // 2 - button_height - button_spacing
            
            # Create main menu buttons with key hints in their text
            start_button = Button(
                screen_width//2 - button_width//2,
                start_y,
                button_width,
                button_height,
                f"1. Start Game" # Added key hint
            )
            settings_button = Button(
                screen_width//2 - button_width//2,
                start_y + button_height + button_spacing,
                button_width,
                button_height,
                f"2. Settings" # Added key hint
            )
            quit_button = Button(
                screen_width//2 - button_width//2,
                start_y + (button_height + button_spacing) * 2,
                button_width,
                button_height,
                f"3. Quit Game" # Added key hint
            )
            
            # Render title
            title_font = pygame.font.Font(None, 72)
            title_text = title_font.render("Mario-like Emotion Platformer", True, BLACK)
            title_rect = title_text.get_rect(center=(screen_width//2, 150))
            
            # Draw everything
            self.screen.fill(MARIO_SKY_BLUE)
            self.screen.blit(title_text, title_rect)
            start_button.draw(self.screen)
            settings_button.draw(self.screen)
            quit_button.draw(self.screen)
            pygame.display.flip() # Update display
            
            # Event handling for main menu
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                
                # Handle keyboard input for menu selection
                if event.type == pygame.KEYDOWN:
                    if event.key == KEY_START:
                        self.game_sounds.stop_music() # Stop menu music when starting game
                        return  # Start the game, exit menu loop
                    elif event.key == KEY_SETTINGS:
                        self.show_settings_menu()  # Show settings menu, then return here
                    elif event.key == KEY_QUIT:
                        self.running = False
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_ESCAPE: # Allow ESC to quit from main menu
                        self.running = False
                        pygame.quit()
                        sys.exit()
                
                # Keep existing mouse hover functionality for visual feedback if desired
                start_button.handle_event(event)
                settings_button.handle_event(event)
        
    def emotion_detection_loop(self):
        """
        Runs in a separate thread to continuously detect emotion
        while the game is running.
        """
        while self.running:
            self.emotion_detector.detect_emotion()
            # Removed time.sleep(0.1) to allow for faster emotion updates.
            # OpenCV's waitKey(1) already provides a small delay.
    
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
        Returns False if player collides with an obstacle (game over),
        unless the player is grown up, in which case the obstacle is removed.
        """
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        
        # Check obstacle collisions
        for obstacle in self.obstacles[:]: # Iterate over a copy to allow safe removal
            obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)
            if player_rect.colliderect(obstacle_rect):
                if self.player.is_grown_up:
                    self.obstacles.remove(obstacle) # Remove obstacle if player is grown up
                    self.score += 50 # Reward for defeating an enemy
                    self.game_sounds.play_stomp_sound() # Play stomp sound
                    print("Enemy defeated!")
                else:
                    self.game_sounds.play_game_over_sound() # Play game over sound
                    return False # Game over if collides with obstacle and not grown up
        
        # Check collectible collisions
        # Iterate over a copy of the list to safely remove items during iteration
        for collectible in self.collectibles[:]: 
            collectible_rect = pygame.Rect(collectible.x, collectible.y, collectible.width, collectible.height)
            if player_rect.colliderect(collectible_rect):
                self.collectibles.remove(collectible) # Remove collected item
                self.score += 10 # Increase score
                self.game_sounds.play_coin_sound() # Play coin sound
        
        return True # No fatal collisions
    
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
        # Create semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128) # 128 out of 255 for transparency
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Draw pause menu text
        title_font = pygame.font.Font(None, 72)
        menu_font = pygame.font.Font(None, 48)
        
        pause_text = title_font.render("PAUSED", True, WHITE)
        continue_text = menu_font.render("Press P to Continue", True, WHITE)
        quit_text = menu_font.render("Press ESC to Quit", True, WHITE)
        
        # Blit text centered on the screen
        self.screen.blit(pause_text, (SCREEN_WIDTH//2 - pause_text.get_width()//2, SCREEN_HEIGHT//2 - 100))
        self.screen.blit(continue_text, (SCREEN_WIDTH//2 - continue_text.get_width()//2, SCREEN_HEIGHT//2))
        self.screen.blit(quit_text, (SCREEN_WIDTH//2 - quit_text.get_width()//2, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip() # Update the display to show the pause menu

    def draw(self):
        """
        Draws all game elements on the screen: background, clouds, platforms,
        obstacles, collectibles, player, and UI elements.
        Also handles game over and pause screen overlays.
        """
        self.draw_background()
        
        # Draw clouds
        for cloud in self.clouds:
            cloud.draw(self.screen)
        
        # Draw platforms
        for platform in self.platforms:
            platform.draw(self.screen)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        # Draw collectibles
        for collectible in self.collectibles:
            collectible.draw(self.screen)
        
        # Draw player
        self.player.draw(self.screen)
        
        # Draw UI elements (emotion, score, distance, FPS)
        emotion_text = self.small_font.render(f"Emotion: {self.emotion_detector.current_emotion}", True, BLACK)
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        distance_text = self.small_font.render(f"Distance: {self.distance//10}m", True, BLACK)
        
        fps = int(self.clock.get_fps())
        fps_text = self.small_font.render(f"FPS: {fps}", True, BLACK)
        self.screen.blit(fps_text, (SCREEN_WIDTH - 100, 10))
        
        self.screen.blit(emotion_text, (10, 10))
        self.screen.blit(score_text, (10, 35))
        self.screen.blit(distance_text, (10, 65))
        
        # Display grow-up status and cooldown
        if self.player.is_grown_up:
            remaining_time = max(0, int(self.player.grow_up_duration - (time.time() - self.player.grow_up_start_time)))
            grow_up_status_text = self.small_font.render(f"GROW UP! ({remaining_time}s remaining)", True, MARIO_RED)
            self.screen.blit(grow_up_status_text, (SCREEN_WIDTH // 2 - grow_up_status_text.get_width() // 2, 10))
        else:
            cooldown_remaining = max(0, int(self.player.grow_up_cooldown - (time.time() - self.player.last_grow_up_time)))
            if cooldown_remaining > 0:
                cooldown_text = self.small_font.render(f"Grow Up Cooldown: {cooldown_remaining}s", True, BLACK)
            else:
                cooldown_text = self.small_font.render("Grow Up Ready!", True, MARIO_GROUND_TOP_GREEN)
            self.screen.blit(cooldown_text, (SCREEN_WIDTH // 2 - cooldown_text.get_width() // 2, 10))


        # Draw instructions based on control method
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            instruction_text = self.small_font.render("SPACE: Jump, RIGHT/D: Move Forward, LEFT/A: Move Backward, ENTER: Grow Up, P: Pause, Others: Stop", True, BLACK)
            self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
        else:
            instruction_text1 = self.small_font.render("NEUTRAL: Stop, HAPPY: Move Forward, SURPRISE: Jump, SAD: Move Backward, ANGRY: Grow Up, P: Pause", True, BLACK)
            self.screen.blit(instruction_text1, (10, SCREEN_HEIGHT - 50))
            if not self.emotion_detector.face_detected:
                instruction_text2 = self.small_font.render("NO FACE DETECTED - Character Stopped! Please look at the camera.", True, MARIO_RED)
            else:
                instruction_text2 = self.small_font.render("Look at the camera and show your emotions!", True, BLACK)
            self.screen.blit(instruction_text2, (10, SCREEN_HEIGHT - 30))
        
        # Game over screen overlay
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128) # Semi-transparent overlay
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font.render("GAME OVER", True, WHITE)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.small_font.render("Press R to restart or ESC to quit", True, WHITE)
            
            # Blit game over text centered
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 60))
            self.screen.blit(final_score_text, (SCREEN_WIDTH//2 - final_score_text.get_width()//2, SCREEN_HEIGHT//2 - 20))
            self.screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2 + 20))
        
        # Show pause menu if game is paused (drawn on top of everything else)
        if self.paused:
            self.show_pause_menu()
        
        pygame.display.flip() # Update the full display Surface to the screen
    
    def restart_game(self):
        """Resets all game elements to their initial state for a new game."""
        self.player = Player(100, 300)
        # Reset initial platforms for a fresh start
        self.platforms = [Platform(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH * 2)]
        self.platforms.append(Platform(SCREEN_WIDTH * 0.8, SCREEN_HEIGHT - 150, 150))
        self.platforms.append(Platform(SCREEN_WIDTH * 1.2, SCREEN_HEIGHT - 250, 200))

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
                    if event.key == pygame.K_ESCAPE:
                        self.running = False # Quit game on ESC
                    elif event.key == pygame.K_r and self.game_over:
                        self.restart_game() # Restart game on 'R' key press if game over
                    elif event.key == pygame.K_p and not self.game_over:
                        self.paused = not self.paused # Toggle pause with P key
                        if self.paused:
                            pygame.mixer.music.pause() # Pause music when game is paused
                        else:
                            pygame.mixer.music.unpause() # Unpause music when game resumes
            
            # Only update game logic if not paused and not game over
            if not self.paused and not self.game_over:
                self.update() # Update game logic
            
            self.draw() # Redraw game elements (including pause/game over overlays if active)
            self.clock.tick(FPS) # Control game speed to target FPS
        
        # Cleanup: release webcam and destroy OpenCV windows when game loop exits
        if hasattr(self.emotion_detector, 'webcam') and self.emotion_detector.webcam:
            self.emotion_detector.webcam.release()
        cv2.destroyAllWindows()
        pygame.quit() # Uninitialize Pygame modules
        sys.exit() # Exit the program

if __name__ == "__main__":
    game = Game()
    game.run()
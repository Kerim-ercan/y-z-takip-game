import cv2
import pygame
import numpy as np
from keras.models import model_from_json
import sys
import threading
import time
import random
import math

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 0.8
JUMP_STRENGTH = -15
PLATFORM_SPEED = 3
PLAYER_SPEED = 4  # Increased player speed for a more dynamic feel

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
BUTTON_HOVER = (100, 100, 100)  # Color for button hover state

class Button:
    def __init__(self, x, y, width, height, text, font_size=48, text_color=WHITE, button_color=MARIO_GROUND_TOP_GREEN):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.text_color = text_color
        self.button_color = button_color
        self.is_hovered = False
        
    def draw(self, screen):
        # Draw button background
        color = BUTTON_HOVER if self.is_hovered else self.button_color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=10)  # Button border
        
        # Draw button text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class EmotionDetector:
    def __init__(self):
        self.current_emotion = 'neutral'
        self.setup_model()
        self.setup_camera()
        
    def setup_model(self):
        try:
            # Load emotion detection model from the provided JSON and H5 files
            json_file = open("emotiondetector.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            # Note: emotiondetector.h5 is expected to be in the same directory for this to work
            self.model.load_weights("emotiondetector.h5") 
            
            # Load Haar cascade for face detection
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_file)
            
            # Define emotion labels
            self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                          4: 'neutral', 5: 'sad', 6: 'surprise'}
            print("Emotion detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using keyboard controls instead...")
            self.model = None
    
    def setup_camera(self):
        # Initialize webcam if model loaded successfully
        if self.model:
            try:
                self.webcam = cv2.VideoCapture(0)
                if not self.webcam.isOpened():
                    print("Camera not available, using keyboard controls")
                    self.webcam = None
            except:
                print("Camera setup failed, using keyboard controls")
                self.webcam = None
    
    def extract_features(self, image):
        # Preprocess image for model prediction
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def detect_emotion(self):
        # If model or webcam not available, use current emotion
        if not self.model or not self.webcam:
            return self.current_emotion
            
        try:
            ret, frame = self.webcam.read()
            if not ret:
                return self.current_emotion
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw rectangle around face
                
                face_img = cv2.resize(face_img, (48, 48)) # Resize face image to 48x48
                img_features = self.extract_features(face_img)
                prediction = self.model.predict(img_features, verbose=0) # Predict emotion
                emotion_label = self.labels[prediction.argmax()] # Get emotion label
                
                cv2.putText(frame, emotion_label, (x-10, y-10), # Display emotion on frame
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                
                self.current_emotion = emotion_label
                break # Process only the first detected face
            
            small_frame = cv2.resize(frame, (200, 150)) # Resize frame for display
            cv2.imshow("Emotion", small_frame) # Show the frame
            cv2.waitKey(1) # Wait for a key press
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
        
        return self.current_emotion

class Player:
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
        
    def update(self, emotion, platforms):
        # Handle emotion-based controls
        if emotion == 'neutral':
            # Stop horizontal movement
            self.vel_x = 0
        elif emotion == 'happy':
            # Move forward
            self.vel_x = PLAYER_SPEED
        elif emotion == 'surprise':
            # Move forward AND jump when surprised
            self.vel_x = PLAYER_SPEED
            if emotion != self.last_emotion and self.jump_count < self.max_jumps:
                # Jump only when emotion changes to surprise (prevents continuous jumping)
                self.vel_y = JUMP_STRENGTH
                self.jump_count += 1
                self.on_ground = False
        elif emotion == 'sad': # New condition for 'sad' emotion
            # Move backward
            self.vel_x = -PLAYER_SPEED
        
        # Store last emotion
        self.last_emotion = emotion
        
        # Apply gravity
        self.vel_y += GRAVITY
        
        # Update position
        self.x += self.vel_x
        self.y += self.vel_y
        
        # Keep player within screen bounds horizontally
        if self.x < 0:
            self.x = 0
        elif self.x + self.width > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH - self.width
        
        # Platform collision
        self.on_ground = False
        for platform in platforms:
            # Check if player is colliding with platform from above
            if (self.x + self.width > platform.x and 
                self.x < platform.x + platform.width and
                self.y + self.height > platform.y and 
                self.y + self.height < platform.y + platform.height + 20 and # Small buffer for collision detection
                self.vel_y > 0): # Only if falling
                
                self.y = platform.y - self.height # Snap player to top of platform
                self.vel_y = 0 # Stop vertical movement
                self.on_ground = True # Player is on ground
                self.jump_count = 0 # Reset jump count
        
        # Screen boundaries (game over if player falls off screen)
        if self.y > SCREEN_HEIGHT:
            return False  # Game over
        
        return True
    
    def draw(self, screen):
        # Draw Mario-like character
        
        # Body (Overalls - Blue)
        pygame.draw.rect(screen, MARIO_BLUE, (self.x + 5, self.y + 15, self.width - 10, self.height - 15), border_radius=3)
        # Shirt (Red)
        pygame.draw.rect(screen, MARIO_RED, (self.x, self.y + 10, self.width, self.height - 20), border_radius=3)
        # Head (Skin color)
        pygame.draw.circle(screen, MARIO_SKIN, (int(self.x + self.width//2), int(self.y + 8)), 10)
        # Hat (Red)
        pygame.draw.rect(screen, MARIO_RED, (self.x + 2, self.y, self.width - 4, 10), border_radius=3)
        pygame.draw.rect(screen, MARIO_RED, (self.x - 5, self.y + 5, 15, 5), border_radius=2) # Hat brim
        # Eyes (Black)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width//2 - 4), int(self.y + 7)), 2)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width//2 + 4), int(self.y + 7)), 2)
        # Mustache (Brown)
        pygame.draw.line(screen, MARIO_GROUND_BROWN, (self.x + self.width//2 - 5, self.y + 12), (self.x + self.width//2 + 5, self.y + 12), 2)
        # Shoes (Brown)
        pygame.draw.rect(screen, MARIO_GROUND_BROWN, (self.x + 2, self.y + self.height - 5, self.width - 4, 5), border_radius=2)

class Platform:
    def __init__(self, x, y, width):
        self.x, self.y, self.width = x, y, width
        self.height = 20
        # Cached Surface creation for performance
        total_height = self.height + 30
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
    
    def update(self, move_platforms=True):
        # Only move platforms when specified (for static world when player stops)
        if move_platforms:
            self.x -= PLATFORM_SPEED

    def draw(self, screen):
        screen.blit(self.cached_surf, (self.x, self.y))

class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        
    def update(self, move_obstacles=True):
        if move_obstacles:
            self.x -= PLATFORM_SPEED
        
    def draw(self, screen):
        # Draw a simple Goomba-like obstacle
        # Body (brown mushroom shape)
        pygame.draw.ellipse(screen, GOOMBA_BROWN, (self.x, self.y, self.width, self.height))
        # Feet (darker brown)
        pygame.draw.rect(screen, GOOMBA_FEET, (self.x + 5, self.y + self.height - 5, self.width - 10, 5))
        # Eyes (white with black pupils)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width * 0.3), int(self.y + self.height * 0.3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.3), int(self.y + self.height * 0.3)), 1)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width * 0.7), int(self.y + self.height * 0.3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width * 0.7), int(self.y + self.height * 0.3)), 1)

class Cloud:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.speed = random.uniform(0.5, 1.5)
        
    def update(self, move_clouds=True):
        if move_clouds:
            self.x -= self.speed
            if self.x < -self.size:
                self.x = SCREEN_WIDTH + random.randint(50, 200)
        
    def draw(self, screen):
        # Draw fluffy cloud
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.size//2)
        pygame.draw.circle(screen, WHITE, (int(self.x + self.size//3), int(self.y)), self.size//3)
        pygame.draw.circle(screen, WHITE, (int(self.x - self.size//3), int(self.y)), self.size//3)

class Collectible:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 20
        self.height = 20
        self.bounce = 0
        
    def update(self, move_collectibles=True):
        if move_collectibles:
            self.x -= PLATFORM_SPEED
        self.bounce += 0.1 # Slower bounce for coin/star
        
    def draw(self, screen):
        # Draw a spinning coin/star
        bounce_y = self.y + math.sin(self.bounce) * 5 # More pronounced bounce
        
        # Draw a star
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
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, 48)
    text = font.render("Loading…", True, (255, 255, 255))
    # center the text
    rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
    screen.blit(text, rect)
    pygame.display.flip()

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Mario-like Emotion Platformer") # Updated title
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False  # Add paused state
        
        # Show loading screen
        show_loading_screen(self.screen)
        # Initialize game objects
        self.player = Player(100, 300)
        self.emotion_detector = EmotionDetector()
        
        # Game objects
        # Initial platforms for a starting area
        # Create a long ground platform at the bottom
        self.platforms = [Platform(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH * 2)] # Long ground platform
        # Add a few initial floating platforms for variety
        self.platforms.append(Platform(SCREEN_WIDTH * 0.8, SCREEN_HEIGHT - 150, 150))
        self.platforms.append(Platform(SCREEN_WIDTH * 1.2, SCREEN_HEIGHT - 250, 200))
        
        self.obstacles = []
        self.clouds = [Cloud(random.randint(0, SCREEN_WIDTH), random.randint(50, 200), random.randint(40, 80)) for _ in range(5)]
        self.collectibles = []
        
        # Game state
        self.score = 0
        self.distance = 0
        self.game_over = False
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Start emotion detection in a separate thread
        if self.emotion_detector.model and self.emotion_detector.webcam:
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop)
            self.emotion_thread.daemon = True # Daemonize thread so it exits with main program
            self.emotion_thread.start()

    def show_main_menu(self):
        # Create menu buttons
        button_width = 300
        button_height = 60
        button_spacing = 20
        start_y = SCREEN_HEIGHT // 2 - button_height
        
        start_button = Button(
            SCREEN_WIDTH//2 - button_width//2,
            start_y,
            button_width,
            button_height,
            "Start Game"
        )
        
        quit_button = Button(
            SCREEN_WIDTH//2 - button_width//2,
            start_y + button_height + button_spacing,
            button_width,
            button_height,
            "Quit Game"
        )
        
        # Title text
        title_font = pygame.font.Font(None, 72)
        title_text = title_font.render("Mario-like Emotion Platformer", True, BLACK)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, 150))
        
        # Menu loop
        while True:
            self.screen.fill(MARIO_SKY_BLUE)
            
            # Draw title
            self.screen.blit(title_text, title_rect)
            
            # Draw and handle buttons
            start_button.draw(self.screen)
            quit_button.draw(self.screen)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                
                if start_button.handle_event(event):
                    return  # Start the game
                if quit_button.handle_event(event):
                    self.running = False
                    pygame.quit()
                    sys.exit()
                
                # Keep keyboard controls as backup
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return  # Start game
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                        pygame.quit()
                        sys.exit()
    
    def emotion_detection_loop(self):
        # Continuously detect emotion while the game is running
        while self.running:
            self.emotion_detector.detect_emotion()
            # Removed time.sleep(0.1) to improve camera FPS and responsiveness
    
    def handle_keyboard_controls(self):
        # Fallback keyboard controls if emotion detection is not available
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.emotion_detector.current_emotion = 'surprise'
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.emotion_detector.current_emotion = 'happy'
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: # Added keyboard control for 'sad'
            self.emotion_detector.current_emotion = 'sad'
        else:
            self.emotion_detector.current_emotion = 'neutral'
    
    def spawn_platform(self):
        # Spawn new platforms as the player progresses
        # Ensure there's always a new platform appearing on the right
        if len(self.platforms) == 0 or self.platforms[-1].x < SCREEN_WIDTH - 100: # Adjust spacing
            x = SCREEN_WIDTH + random.randint(0, 100) # Randomize starting x slightly
            
            # Decide if it's a ground extension or a floating platform
            if random.random() < 0.6: # 60% chance for ground extension
                y = SCREEN_HEIGHT - 50 # Consistent ground level
                width = random.randint(150, 300) # Longer ground segments
            else: # 40% chance for floating platform
                y = random.randint(SCREEN_HEIGHT - 250, SCREEN_HEIGHT - 100) # Floating platforms above ground
                width = random.randint(80, 200) # Shorter floating platforms
            
            self.platforms.append(Platform(x, y, width))
    
    def spawn_obstacle(self):
        # Spawn obstacles on existing platforms
        for platform in self.platforms:
            # Only spawn on platforms that are visible and not too close to the edge
            if (platform.x > SCREEN_WIDTH - 200 and platform.x < SCREEN_WIDTH - 100 and
                random.random() < 0.01): # Lower chance to spawn obstacles
                self.obstacles.append(Obstacle(platform.x + random.randint(20, platform.width - 50), 
                                             platform.y - 30)) # Position above platform
    
    def spawn_collectible(self):
        # Spawn collectibles on existing platforms
        for platform in self.platforms:
            if (platform.x > SCREEN_WIDTH - 250 and platform.x < SCREEN_WIDTH - 150 and
                random.random() < 0.05): # Higher chance for collectibles
                self.collectibles.append(Collectible(platform.x + random.randint(10, platform.width - 40), 
                                                   platform.y - 40)) # Position above platform
    
    def check_collisions(self):
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)
            if player_rect.colliderect(obstacle_rect):
                return False # Game over if collides with obstacle
        
        # Check collectible collisions
        for collectible in self.collectibles[:]: # Iterate over a copy to allow removal
            collectible_rect = pygame.Rect(collectible.x, collectible.y, collectible.width, collectible.height)
            if player_rect.colliderect(collectible_rect):
                self.collectibles.remove(collectible) # Remove collected item
                self.score += 10 # Increase score
        
        return True
    
    def update(self):
        if self.game_over:
            return
        
        # Handle keyboard as backup if emotion detection is not active
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            self.handle_keyboard_controls()
        
        current_emotion = self.emotion_detector.current_emotion
        
        # Update player position and check for game over condition
        if not self.player.update(current_emotion, self.platforms):
            self.game_over = True
            return
        
        # Check for collisions with obstacles and collectibles
        if not self.check_collisions():
            self.game_over = True
            return
        
        # Determine if the game world should scroll (when player is moving forward)
        # The world should move forward if happy/surprise, backward if sad, and stop if neutral.
        # This logic needs to be adjusted based on the new 'sad' movement.
        # If the player is moving backward (sad), the world should effectively move forward
        # relative to the player's movement, but the platforms themselves will move
        # based on the player's velocity.
        
        # Let's simplify: if player.vel_x is positive, world moves backward (normal scrolling)
        # If player.vel_x is negative, world moves forward (reverse scrolling)
        # If player.vel_x is zero, world stands still.
        
        # In a side-scroller, the world usually moves opposite to the player's intended direction.
        # If player moves right, world moves left. If player moves left, world moves right.
        # However, since the player's x position is capped at screen bounds, we need to
        # decide if the *platforms* themselves should move.
        
        # For a Mario-like game, the world scrolls *left* when Mario moves *right*.
        # If Mario moves *left*, the world *stops* scrolling left, or scrolls *right* if he's
        # far enough left on the screen.
        
        # Let's adjust the world movement based on player's horizontal velocity.
        # If player is moving right (happy/surprise), platforms move left (normal scrolling)
        # If player is moving left (sad), platforms move right (reverse scrolling)
        # If player is stationary (neutral), platforms stop.

        # The current PLATFORM_SPEED is always subtracting from platform.x.
        # We need to make it dynamic based on player's vel_x.

        # Calculate effective platform movement speed
        effective_platform_speed = 0
        if current_emotion == 'happy' or current_emotion == 'surprise':
            effective_platform_speed = -PLATFORM_SPEED # Platforms move left
        elif current_emotion == 'sad':
            effective_platform_speed = PLATFORM_SPEED # Platforms move right (to simulate player going left)
        
        # Update platforms
        for platform in self.platforms[:]:
            platform.x += effective_platform_speed # Apply dynamic speed
            if platform.x + platform.width < 0 and effective_platform_speed < 0: # Remove if off left and moving left
                self.platforms.remove(platform)
            elif platform.x > SCREEN_WIDTH and effective_platform_speed > 0: # Remove if off right and moving right
                self.platforms.remove(platform)
        
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle.x += effective_platform_speed
            if obstacle.x + obstacle.width < 0 and effective_platform_speed < 0:
                self.obstacles.remove(obstacle)
            elif obstacle.x > SCREEN_WIDTH and effective_platform_speed > 0:
                self.obstacles.remove(obstacle)
        
        # Update collectibles
        for collectible in self.collectibles[:]:
            collectible.x += effective_platform_speed
            if collectible.x + collectible.width < 0 and effective_platform_speed < 0:
                self.collectibles.remove(collectible)
            elif collectible.x > SCREEN_WIDTH and effective_platform_speed > 0:
                self.collectibles.remove(collectible)
        
        # Update clouds (clouds should always move left, but perhaps slower)
        for cloud in self.clouds:
            # Clouds always move left, regardless of player direction, but slower than platforms
            cloud.x -= cloud.speed * 0.5 # Slower cloud movement
            if cloud.x < -cloud.size:
                cloud.x = SCREEN_WIDTH + random.randint(50, 200)
        
        # Spawn new objects only when moving right (normal scrolling)
        # Spawning logic needs to consider if the world is moving forward or backward
        # For now, let's keep spawning only when player is moving forward (happy/surprise)
        if current_emotion == 'happy' or current_emotion == 'surprise':
            self.spawn_platform()
            self.spawn_obstacle()
            self.spawn_collectible()
            
            # Update game state (distance and score)
            self.distance += PLATFORM_SPEED
            if self.distance % 100 == 0: # Award score for distance traveled
                self.score += 1
        elif current_emotion == 'sad':
            # If moving backward, we might want to "un-spawn" or ensure no new objects appear
            # or even have a limited "backtrack" area. For simplicity, no new spawns when sad.
            pass # No new spawns when moving backward
    
    def draw_background(self):
        # Fill the background with Mario-like sky blue
        self.screen.fill(MARIO_SKY_BLUE)
    
    def show_pause_menu(self):
        # Create semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Draw pause menu text
        title_font = pygame.font.Font(None, 72)
        menu_font = pygame.font.Font(None, 48)
        
        pause_text = title_font.render("PAUSED", True, WHITE)
        continue_text = menu_font.render("Press P to Continue", True, WHITE)
        quit_text = menu_font.render("Press ESC to Quit", True, WHITE)
        
        self.screen.blit(pause_text, (SCREEN_WIDTH//2 - pause_text.get_width()//2, SCREEN_HEIGHT//2 - 100))
        self.screen.blit(continue_text, (SCREEN_WIDTH//2 - continue_text.get_width()//2, SCREEN_HEIGHT//2))
        self.screen.blit(quit_text, (SCREEN_WIDTH//2 - quit_text.get_width()//2, SCREEN_HEIGHT//2 + 50))
        
        pygame.display.flip()

    def draw(self):
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
        
        # Draw instructions based on control method
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            instruction_text = self.small_font.render("SPACE: Jump, RIGHT/D: Move Forward, LEFT/A: Move Backward, P: Pause, Others: Stop", True, BLACK)
            self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
        else:
            instruction_text1 = self.small_font.render("NEUTRAL: Stop, HAPPY: Move Forward, SURPRISE: Jump, SAD: Move Backward, P: Pause", True, BLACK)
            self.screen.blit(instruction_text1, (10, SCREEN_HEIGHT - 50))
            instruction_text2 = self.small_font.render("Look at the camera and show your emotions!", True, BLACK)
            self.screen.blit(instruction_text2, (10, SCREEN_HEIGHT - 30))
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128) # Semi-transparent overlay
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font.render("GAME OVER", True, WHITE)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.small_font.render("Press R to restart or ESC to quit", True, WHITE)
            
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 60))
            self.screen.blit(final_score_text, (SCREEN_WIDTH//2 - final_score_text.get_width()//2, SCREEN_HEIGHT//2 - 20))
            self.screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, SCREEN_HEIGHT//2 + 20))
        
        # Show pause menu if game is paused
        if self.paused:
            self.show_pause_menu()
        
        pygame.display.flip() # Update the full display Surface to the screen
    
    def restart_game(self):
        # Reset all game elements to their initial state
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
    
    def run(self):
        self.show_main_menu() # Show main menu before starting the game loop
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.restart_game() # Restart game on 'R' key press if game over
                    elif event.key == pygame.K_p and not self.game_over:  # Toggle pause with P key
                        self.paused = not self.paused
            
            if not self.paused and not self.game_over:  # Only update game if not paused and not game over
                self.update() # Update game logic
            
            self.draw() # Redraw game elements
            self.clock.tick(FPS) # Control game speed
        
        # Cleanup: release webcam and destroy OpenCV windows
        if hasattr(self.emotion_detector, 'webcam') and self.emotion_detector.webcam:
            self.emotion_detector.webcam.release()
        cv2.destroyAllWindows()
        pygame.quit() # Uninitialize Pygame modules
        sys.exit() # Exit the program

if __name__ == "__main__":
    game = Game()
    game.run()

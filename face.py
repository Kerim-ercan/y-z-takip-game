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
PLAYER_SPEED = 0.5  # New constant for player movement

# Colors
SKY_BLUE = (135, 206, 250)
TEAL = (0, 128, 128)
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
YELLOW = (255, 215, 0)
PINK = (255, 192, 203)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)

class EmotionDetector:
    def __init__(self):
        self.current_emotion = 'neutral'
        self.setup_model()
        self.setup_camera()
        
    def setup_model(self):
        try:
            json_file = open("emotiondetector.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights("emotiondetector.h5")
            
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(haar_file)
            
            self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                          4: 'neutral', 5: 'sad', 6: 'surprise'}
            print("Emotion detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using keyboard controls instead...")
            self.model = None
    
    def setup_camera(self):
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
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def detect_emotion(self):
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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                face_img = cv2.resize(face_img, (48, 48))
                img_features = self.extract_features(face_img)
                prediction = self.model.predict(img_features, verbose=0)
                emotion_label = self.labels[prediction.argmax()]
                
                cv2.putText(frame, emotion_label, (x-10, y-10),
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                
                self.current_emotion = emotion_label
                break
            
            small_frame = cv2.resize(frame, (200, 150))
            cv2.imshow("Emotion", small_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
        
        return self.current_emotion

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 35
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
        elif emotion == 'surprise' and emotion != self.last_emotion and self.jump_count < self.max_jumps:
            # Jump only when emotion changes to surprise (prevents continuous jumping)
            self.vel_y = JUMP_STRENGTH
            self.jump_count += 1
            self.on_ground = False
        
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
            if (self.x + self.width > platform.x and 
                self.x < platform.x + platform.width and
                self.y + self.height > platform.y and 
                self.y + self.height < platform.y + platform.height + 20 and
                self.vel_y > 0):
                
                self.y = platform.y - self.height
                self.vel_y = 0
                self.on_ground = True
                self.jump_count = 0
        
        # Screen boundaries
        if self.y > SCREEN_HEIGHT:
            return False  # Game over
        
        return True
    
    def draw(self, screen):
        # Draw cute character (like the bird in the image)
        # Body
        pygame.draw.ellipse(screen, WHITE, (self.x, self.y + 10, self.width, self.height - 15))
        # Head
        pygame.draw.circle(screen, WHITE, (int(self.x + self.width//2), int(self.y + 8)), 12)
        # Beak
        points = [(self.x + self.width//2 + 8, self.y + 8), 
                 (self.x + self.width//2 + 15, self.y + 8),
                 (self.x + self.width//2 + 12, self.y + 12)]
        pygame.draw.polygon(screen, YELLOW, points)
        # Eyes
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width//2 - 3), int(self.y + 5)), 2)
        pygame.draw.circle(screen, BLACK, (int(self.x + self.width//2 + 3), int(self.y + 5)), 2)
        # Wing
        pygame.draw.ellipse(screen, PINK, (self.x + 5, self.y + 12, 15, 8))

class Platform:
    def __init__(self, x, y, width):
        self.x, self.y, self.width = x, y, width
        self.height = 20
        # Cached Surface creation
        total_height = self.height + 30
        self.cached_surf = pygame.Surface((self.width, total_height), pygame.SRCALPHA)

        # Draw platform on cached surface
        # Top ground (green)
        pygame.draw.rect(self.cached_surf, GREEN, (0, 0, self.width, self.height))
        # Bottom layer (darker)
        pygame.draw.rect(self.cached_surf, DARK_GREEN, (0, self.height, self.width, 30))
        # Grass effects
        for i in range(0, self.width, 10):
            pygame.draw.line(
                self.cached_surf, DARK_GREEN,
                (i, 0), (i+3, -5), 2
            )
    
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
        self.width = 25
        self.height = 40
        
    def update(self, move_obstacles=True):
        if move_obstacles:
            self.x -= PLATFORM_SPEED
        
    def draw(self, screen):
        # Draw spike obstacle
        points = [(self.x, self.y + self.height),
                 (self.x + self.width//2, self.y),
                 (self.x + self.width, self.y + self.height)]
        pygame.draw.polygon(screen, BLACK, points)

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
        self.bounce += 0.2
        
    def draw(self, screen):
        # Draw musical note (like in the image)
        bounce_y = self.y + math.sin(self.bounce) * 3
        pygame.draw.circle(screen, YELLOW, (int(self.x + 10), int(bounce_y + 15)), 8)
        pygame.draw.rect(screen, BLACK, (self.x + 17, bounce_y, 3, 20))
        pygame.draw.arc(screen, BLACK, (self.x + 15, bounce_y - 5, 10, 10), 0, math.pi, 3)

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
        pygame.display.set_caption("Emotion Platform Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Show loading screen
        show_loading_screen(self.screen)
        # Initialize game objects
        self.player = Player(100, 300)
        self.emotion_detector = EmotionDetector()
        
        # Game objects
        self.platforms = [Platform(0, 400, 200), Platform(250, 350, 150), Platform(450, 300, 200)]
        self.obstacles = []
        self.clouds = [Cloud(random.randint(0, SCREEN_WIDTH), random.randint(50, 200), random.randint(40, 80)) for _ in range(5)]
        self.collectibles = []
        
        # Game state
        self.score = 0
        self.distance = 0
        self.game_over = False
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Timers
        self.platform_timer = 0
        self.obstacle_timer = 0
        self.collectible_timer = 0
        
        # Start emotion detection
        if self.emotion_detector.model and self.emotion_detector.webcam:
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop)
            self.emotion_thread.daemon = True
            self.emotion_thread.start()

    def show_main_menu(self):
        title_font = pygame.font.Font(None, 72)
        button_font = pygame.font.Font(None, 48)

        while True:
            self.screen.fill(SKY_BLUE)

            title_text = title_font.render("Emotion Platform Game", True, BLACK)
            start_text = button_font.render("Press ENTER to Start", True, DARK_GREEN)
            quit_text = button_font.render("Press ESC to Quit", True, DARK_GREEN)

            self.screen.blit(title_text, (SCREEN_WIDTH//2 - title_text.get_width()//2, 150))
            self.screen.blit(start_text, (SCREEN_WIDTH//2 - start_text.get_width()//2, 300))
            self.screen.blit(quit_text, (SCREEN_WIDTH//2 - quit_text.get_width()//2, 370))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                        pygame.quit()
                        sys.exit()
    
    def emotion_detection_loop(self):
        while self.running:
            self.emotion_detector.detect_emotion()
            time.sleep(0.1)
    
    def handle_keyboard_controls(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.emotion_detector.current_emotion = 'surprise'
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.emotion_detector.current_emotion = 'happy'
        else:
            self.emotion_detector.current_emotion = 'neutral'
    
    def spawn_platform(self):
        if len(self.platforms) == 0 or self.platforms[-1].x < SCREEN_WIDTH - 200:
            x = SCREEN_WIDTH
            y = random.randint(200, 500)
            width = random.randint(100, 200)
            self.platforms.append(Platform(x, y, width))
    
    def spawn_obstacle(self):
        for platform in self.platforms:
            if (platform.x > SCREEN_WIDTH - 100 and platform.x < SCREEN_WIDTH - 50 and
                random.random() < 0.3):
                self.obstacles.append(Obstacle(platform.x + random.randint(20, platform.width - 40), 
                                             platform.y - 40))
    
    def spawn_collectible(self):
        for platform in self.platforms:
            if (platform.x > SCREEN_WIDTH - 150 and platform.x < SCREEN_WIDTH - 100 and
                random.random() < 0.2):
                self.collectibles.append(Collectible(platform.x + random.randint(10, platform.width - 30), 
                                                   platform.y - 30))
    
    def check_collisions(self):
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)
            if player_rect.colliderect(obstacle_rect):
                return False
        
        # Check collectible collisions
        for collectible in self.collectibles[:]:
            collectible_rect = pygame.Rect(collectible.x, collectible.y, collectible.width, collectible.height)
            if player_rect.colliderect(collectible_rect):
                self.collectibles.remove(collectible)
                self.score += 10
        
        return True
    
    def update(self):
        if self.game_over:
            return
        
        # Handle keyboard as backup
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            self.handle_keyboard_controls()
        
        current_emotion = self.emotion_detector.current_emotion
        
        # Update player
        if not self.player.update(current_emotion, self.platforms):
            self.game_over = True
            return
        
        # Check collisions
        if not self.check_collisions():
            self.game_over = True
            return
        
        # Determine if world should move (only when player is moving forward)
        world_should_move = (current_emotion == 'happy')
        
        # Update platforms
        for platform in self.platforms[:]:
            platform.update(move_platforms=world_should_move)
            if platform.x + platform.width < 0:
                self.platforms.remove(platform)
        
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update(move_obstacles=world_should_move)
            if obstacle.x + obstacle.width < 0:
                self.obstacles.remove(obstacle)
        
        # Update collectibles
        for collectible in self.collectibles[:]:
            collectible.update(move_collectibles=world_should_move)
            if collectible.x + collectible.width < 0:
                self.collectibles.remove(collectible)
        
        # Update clouds
        for cloud in self.clouds:
            cloud.update(move_clouds=world_should_move)
        
        # Spawn new objects only when moving
        if world_should_move:
            self.spawn_platform()
            self.spawn_obstacle()
            self.spawn_collectible()
            
            # Update game state
            self.distance += PLATFORM_SPEED
            if self.distance % 100 == 0:
                self.score += 1
    
    def draw_background(self):
        # Gradient sky
        for y in range(SCREEN_HEIGHT):
            color_ratio = y / SCREEN_HEIGHT
            r = int(135 * (1 - color_ratio) + 0 * color_ratio)
            g = int(206 * (1 - color_ratio) + 128 * color_ratio)
            b = int(250 * (1 - color_ratio) + 128 * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
    
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
        
        # Draw UI
        emotion_text = self.small_font.render(f"Emotion: {self.emotion_detector.current_emotion}", True, WHITE)
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        distance_text = self.small_font.render(f"Distance: {self.distance//10}m", True, WHITE)
        
        fps = int(self.clock.get_fps())
        fps_text = self.small_font.render(f"FPS: {fps}", True, WHITE)
        self.screen.blit(fps_text, (SCREEN_WIDTH - 100, 10))
        
        self.screen.blit(emotion_text, (10, 10))
        self.screen.blit(score_text, (10, 35))
        self.screen.blit(distance_text, (10, 65))
        
        # Draw instructions
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            instruction_text = self.small_font.render("SPACE: Jump, RIGHT/D: Move, Others: Stop", True, WHITE)
            self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
        else:
            instruction_text1 = self.small_font.render("NEUTRAL: Stop, HAPPY: Move Forward, SURPRISE: Jump", True, WHITE)
            self.screen.blit(instruction_text1, (10, SCREEN_HEIGHT - 50))
            instruction_text2 = self.small_font.render("Look at the camera and show your emotions!", True, WHITE)
            self.screen.blit(instruction_text2, (10, SCREEN_HEIGHT - 30))
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font.render("GAME OVER", True, WHITE)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.small_font.render("Press R to restart or ESC to quit", True, WHITE)
            
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 60))
            self.screen.blit(final_score_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 20))
            self.screen.blit(restart_text, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2 + 20))
        
        pygame.display.flip()
    
    def restart_game(self):
        self.player = Player(100, 300)
        self.platforms = [Platform(0, 400, 200), Platform(250, 350, 150), Platform(450, 300, 200)]
        self.obstacles = []
        self.collectibles = []
        self.score = 0
        self.distance = 0
        self.game_over = False
    
    def run(self):
        self.show_main_menu()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.restart_game()
            
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        # Cleanup
        if hasattr(self.emotion_detector, 'webcam') and self.emotion_detector.webcam:
            self.emotion_detector.webcam.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
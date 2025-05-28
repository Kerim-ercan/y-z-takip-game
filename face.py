import cv2
import pygame
import numpy as np
from keras.models import model_from_json
import sys
import threading
import time

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 0.8
JUMP_STRENGTH = -15

# Colors
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
YELLOW = (255, 255, 100)
BLACK = (0, 0, 0)

class EmotionDetector:
    def __init__(self):
        self.current_emotion = 'neutral'
        self.setup_model()
        self.setup_camera()
        
    def setup_model(self):
        try:
            # Load emotion detection model
            json_file = open("emotiondetector.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights("emotiondetector.h5")
            
            # Setup face detection
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
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
                
                self.current_emotion = emotion_label
                break
            
            # Show camera feed (small window)
            small_frame = cv2.resize(frame, (320, 240))
            cv2.imshow("Emotion Detection", small_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
        
        return self.current_emotion

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 60
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 5
        self.on_ground = False
        self.color = BLUE
        
    def update(self, emotion):
        # Handle emotion-based movement
        if emotion == 'happy':
            self.vel_x = self.speed  # Move forward
            self.color = YELLOW
        elif emotion == 'sad':
            self.vel_x = -self.speed  # Move backward
            self.color = BLUE
        elif emotion == 'surprise':
            if self.on_ground:  # Jump
                self.vel_y = JUMP_STRENGTH
                self.on_ground = False
            self.color = GREEN
        elif emotion == 'angry':
            self.vel_x = self.speed * 1.5  # Move faster forward
            self.color = RED
        else:  # neutral, fear, disgust
            self.vel_x *= 0.8  # Slow down
            self.color = BLUE
        
        # Apply gravity
        if not self.on_ground:
            self.vel_y += GRAVITY
        
        # Update position
        self.x += self.vel_x
        self.y += self.vel_y
        
        # Keep player on screen horizontally
        if self.x < 0:
            self.x = 0
        elif self.x > SCREEN_WIDTH - self.width:
            self.x = SCREEN_WIDTH - self.width
        
        # Ground collision
        if self.y >= SCREEN_HEIGHT - 100 - self.height:
            self.y = SCREEN_HEIGHT - 100 - self.height
            self.vel_y = 0
            self.on_ground = True
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        # Draw eyes
        pygame.draw.circle(screen, BLACK, (int(self.x + 15), int(self.y + 15)), 5)
        pygame.draw.circle(screen, BLACK, (int(self.x + 35), int(self.y + 15)), 5)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Emotion-Controlled Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize game objects
        self.player = Player(100, SCREEN_HEIGHT - 160)
        self.emotion_detector = EmotionDetector()
        
        # Game variables
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        
        # Start emotion detection in separate thread
        if self.emotion_detector.model and self.emotion_detector.webcam:
            self.emotion_thread = threading.Thread(target=self.emotion_detection_loop)
            self.emotion_thread.daemon = True
            self.emotion_thread.start()
    
    def emotion_detection_loop(self):
        while self.running:
            self.emotion_detector.detect_emotion()
            time.sleep(0.1)  # Limit detection rate
    
    def handle_keyboard_controls(self):
        # Fallback keyboard controls if emotion detection fails
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.emotion_detector.current_emotion = 'sad'
        elif keys[pygame.K_RIGHT]:
            self.emotion_detector.current_emotion = 'happy'
        elif keys[pygame.K_SPACE]:
            self.emotion_detector.current_emotion = 'surprise'
        elif keys[pygame.K_UP]:
            self.emotion_detector.current_emotion = 'angry'
        else:
            self.emotion_detector.current_emotion = 'neutral'
    
    def update(self):
        # Handle keyboard as backup
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            self.handle_keyboard_controls()
        
        # Update player based on current emotion
        self.player.update(self.emotion_detector.current_emotion)
        
        # Update score based on movement
        if self.emotion_detector.current_emotion in ['happy', 'angry']:
            self.score += 1
    
    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw ground
        pygame.draw.rect(self.screen, GREEN, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
        
        # Draw player
        self.player.draw(self.screen)
        
        # Draw UI
        emotion_text = self.font.render(f"Emotion: {self.emotion_detector.current_emotion}", 
                                       True, BLACK)
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        
        self.screen.blit(emotion_text, (10, 10))
        self.screen.blit(score_text, (10, 50))
        
        # Draw instructions
        if not self.emotion_detector.model or not self.emotion_detector.webcam:
            instruction_text = pygame.font.Font(None, 24).render(
                "Keyboard: Left=Sad, Right=Happy, Space=Surprise, Up=Angry", True, BLACK)
            self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
        else:
            instruction_text = pygame.font.Font(None, 24).render(
                "Control with your emotions! Happy=Forward, Sad=Back, Surprise=Jump", True, BLACK)
            self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
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
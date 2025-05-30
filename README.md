# Emotion Platformer Game

## 🎮 Description

Emotion Platformer is a fun, Mario-like 2D platformer game where you control the player character using your facial expressions! Navigate through a dynamic world, collect stars, stomp on enemies, and try to achieve the highest score by running as far as you can. If you prefer, or if a webcam isn't available, you can always switch to traditional keyboard controls.

The game uses a Keras-based neural network model to detect your emotions (Happy, Sad, Surprise, Angry, Neutral) via your webcam, translating them into in-game actions.

## ✨ Features

* **Emotion-Based Controls:**
    * **Happy face:** Move Right
    * **Surprise face:** Jump (show the expression again mid-air for a double jump)
    * **Sad face:** Move Left
    * **Angry face:** Activate "Grow Up" mode! Become larger and stomp any enemy. This ability has a cooldown.
    * **Neutral face:** Stop movement
* **Keyboard Controls Fallback:**
    * **Right Arrow / D:** Move Right
    * **Left Arrow / A:** Move Left
    * **Spacebar:** Jump (press again mid-air for a double jump)
    * **Enter:** Activate "Grow Up" mode
* **Dynamic Gameplay:**
    * Endless, procedurally generated world with platforms.
    * Enemies:
        * **Goombas (Brown Mushrooms):** Can be stomped to defeat.
        * **Koopas (Green Turtles):** Moving enemies; can only be stomped when the player is in "Grow Up" mode.
    * Collectibles: Stars to boost your score.
    * Scoring system based on distance traveled and collectibles gathered.
* **Player Abilities:**
    * Double Jump
    * "Grow Up" Power: Temporary size increase and ability to defeat any enemy by touch.
* **Engaging UI:**
    * Real-time display of detected emotion.
    * Live webcam feed integrated into the game screen.
    * HUD displaying Score, Distance, and current FPS.
    * Visual timer and status for the "Grow Up" ability and its cooldown.
    * On-screen control reminders.
* **Interactive Menus:**
    * **Main Menu:** Start Game, How to Play, Settings, Quit.
    * **Settings Menu:** Adjust screen resolution (Small, Medium, Large, Fullscreen) and sound volumes (Master, Music, SFX).
    * **How to Play Screen:** Detailed instructions for both emotion and keyboard controls.
    * **Pause Menu:** Resume, or return to the Main Menu.
    * **Game Over Screen:** Shows final score with options to restart or quit.
* **Sound & Music:**
    * Background music and various sound effects (jump, coin collection, enemy stomp, power-up, game over).
    * Adjustable volume levels, saved across sessions.
* **Customization:**
    * Screen resolution settings are saved in `game_settings.json`.
    * Volume settings are saved in `game_settings.json`.
* **Caution for Glasses:** An initial caution screen advises players to play without glasses for optimal emotion detection performance.

## 🕹️ How to Play

### Emotion Controls (Webcam Active)

1.  Ensure your webcam is connected and has adequate lighting.
2.  Keep your face clearly visible in the webcam feed shown on screen.
3.  **Move Right:** Make a **Happy** face.
4.  **Jump:** Show a **Surprised** face. (Show it again in mid-air for a double jump).
5.  **Move Left:** Make a **Sad** face.
6.  **Grow Up/Power:** Show an **Angry** face. You'll become larger and can defeat any enemy by touching them. This has a duration and a cooldown period.
7.  **Stop:** Maintain a **Neutral** face.
8.  **No Face Detected:** If the game can't see your face, Mario will stop.

### Keyboard Controls

If emotion detection is disabled or as a preference:

1.  **Move Right:** Press the **Right Arrow** key or **D**.
2.  **Move Left:** Press the **Left Arrow** key or **A**.
3.  **Jump:** Press the **Spacebar**. (Press again in mid-air for a double jump).
4.  **Grow Up/Power:** Press the **Enter** key.

### General Gameplay

* **Objective:** Run as far as you can, collect stars for points, and avoid or stomp on enemies.
* **Stomping Enemies:**
    * **Goombas (Brown Mushrooms):** Jump on their heads to defeat them.
    * **Koopas (Green Turtles):** These moving enemies can only be defeated by stomping when you are in "Grow Up" mode.
* **Falling:** Don't fall off the platforms, or it's game over!
* **Pause:** Press **P** to pause or unpause the game.
* **Main Menu/Quit:** Press **ESC** during gameplay to pause and access the option to return to the main menu, or from the game over screen to quit.
* **Restart:** Press **R** on the Game Over screen to play again.

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    Ensure you have Python 3 installed. Then, install the required libraries:
    ```bash
    pip install pygame opencv-python numpy tensorflow keras scikit-learn pandas tqdm
    ```
    *(Note: `pandas`, `scikit-learn`, and `tqdm` are primarily for the `face-detection-for-game.ipynb` notebook but are listed as they appear in imports.)*

3.  **Required Files:**
    Make sure the following files and folders are present in the main project directory:
    * `face.py` (the main game script)
    * `emotiondetector.json` (model architecture)
    * `emotiondetector.h5` (**CRUCIAL:** This file contains the pre-trained model weights. It's loaded by `face.py` and is essential for emotion detection. You'll need to provide this file.)
    * `game_settings.json` (will be created/updated by the game for settings)
    * `pixel_font.ttf` (the custom font file for the game's UI)
    * A `sounds/` directory containing:
        * `game_music.mp3`
        * `jump.mp3`
        * `coin.mp3`
        * `game_over.mp3`
        * `stomp.mp3`
        * `power_up.mp3`
    * The `haarcascade_frontalface_default.xml` file for face detection is usually included with OpenCV. The game loads it from `cv2.data.haarcascades`. If you encounter issues, ensure your OpenCV installation is correct.

4.  **Run the game:**
    ```bash
    python face.py
    ```

## 📂 Project Files

* `face.py`: The main Python script that runs the game. It includes game logic, Pygame initialization, emotion integration, and UI management.
* `emotiondetector.json`: Contains the architecture of the Keras sequential model used for emotion detection.
* `emotiondetector.h5`: **(Not provided in this request, but essential)** Contains the trained weights for the emotion detection model. This file must be present in the same directory as `face.py` for the emotion detection feature to work.
* `face-detection-for-game.ipynb`: A Jupyter Notebook detailing the process of training the emotion detection model. This includes data loading, preprocessing, model creation, training, and evaluation.
* `game_settings.json`: Stores user preferences such as screen size and audio volume levels. Automatically created and updated by the game.
* `pixel_font.ttf`: **(Assumed name)** The custom pixel-art font file used for text rendering in the game.
* `sounds/` (directory): **(Assumed)** This directory should contain all the audio files (music and sound effects) used in the game.

## 🚀 Potential Future Enhancements

* More enemy types with different behaviors.
* Additional power-ups and player abilities.
* Level progression with increasing difficulty.
* Leaderboard system.
* More refined emotion detection model or options for model retraining.
* Customizable keyboard controls.

---

Feel free to modify any part of this to better suit your project's specifics!

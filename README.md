# Racecar AI

A top-down racing game with neural network AI that learns to drive through genetic evolution. Draw your own tracks, drive them manually, or watch AI cars teach themselves to race.

Built with Python and Pygame. Uses the [Catppuccin Macchiato](https://github.com/catppuccin/catppuccin) color palette.

## Features

- **Draw Mode** - Paint custom race tracks with a brush tool and set a start position/direction
- **Drive Mode** - Take manual control of a car with drift physics
- **AI Mode** - Watch a population of 50 neural network-controlled cars evolve to navigate your track

### How the AI Works

Each car is controlled by a small feedforward neural network (8 inputs, 10 hidden neurons, 2 outputs). The inputs are 7 raycasted distance sensors spread across a 180-degree arc in front of the car, plus the car's current speed. The outputs control throttle and steering.

A genetic algorithm evolves the population across generations using tournament selection, crossover, and mutation, with the top 2 performers carried forward via elitism.

## Setup

### Windows (Clean Install)

1. Download or clone this repository
2. Double-click **`install_and_run.bat`**
   - If Python is not installed, the script will download and install Python 3.12 automatically
   - After Python installs, **close the window and run the script again** so the PATH update takes effect
   - On the second run, it will install dependencies (pygame, numpy) and launch the game

### Windows (Manual)

1. Install [Python 3.12+](https://www.python.org/downloads/) — check **"Add Python to PATH"** during installation
2. Open a terminal in the project folder and run:
   ```
   pip install -r requirements.txt
   ```
3. Launch the game:
   ```
   python main.py
   ```

### Linux

1. Install Python 3 and pip if not already available:
   ```
   sudo apt install python3 python3-pip
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the game:
   ```
   python3 main.py
   ```

## Controls

### Mode Switching

| Key | Action |
|-----|--------|
| `1` | Draw Mode |
| `2` | Drive Mode |
| `3` | AI Mode |
| `Esc` / `Q` | Quit |

### Draw Mode

| Input | Action |
|-------|--------|
| Left click | Paint track (drivable surface) |
| Right click | Erase (back to wall) |
| Scroll wheel | Change brush size |
| Shift + click & drag | Place start position and direction |
| `C` | Clear track |
| `Ctrl+S` | Save track (prompts for a name) |
| `Ctrl+L` | Load track (shows numbered list) |

### Drive Mode

| Input | Action |
|-------|--------|
| Arrow keys / WASD | Steer and accelerate |
| `R` | Reset car to start |

### AI Mode

| Input | Action |
|-------|--------|
| `R` | Restart evolution |
| `+` / `-` | Increase / decrease simulation speed (1x–10x) |
| `B` | Save the best brain to file |

## Project Structure

```
racecar_ai/
├── main.py              # Game loop, modes, rendering
├── car.py               # Car physics, raycasting, collision, drift
├── track.py             # Track drawing, saving/loading (PNG + JSON)
├── neural_network.py    # Feedforward NN and genetic algorithm
├── colors.py            # Catppuccin Macchiato color palette
├── tracks/              # Saved tracks and brain files
├── requirements.txt     # Python dependencies
└── install_and_run.bat  # Windows one-click installer
```

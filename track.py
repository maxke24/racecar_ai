import pygame
import numpy as np
import json
import os
from collections import deque
from colors import BASE, SURFACE0, LAVENDER, RED, GREEN, OVERLAY0, SURFACE2


TRACK_DIR = os.path.join(os.path.dirname(__file__), "tracks")


class Track:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # The track surface: Surface0 = drivable, Base = wall (used for collision)
        self.surface = pygame.Surface((width, height))
        self.surface.fill(BASE)

        # Drawing state
        self.drawing = False
        self.erasing = False
        self.brush_size = 40
        self.points = []

        # Start/finish position (lower on screen by default)
        self.start_x = width // 2
        self.start_y = height * 3 // 4
        self.start_angle = 0
        self.placing_start = False

        # Checkpoint system (populated by generate_checkpoints)
        self.checkpoints = []
        self.distance_field = None

    def start_draw(self, pos, erase=False):
        self.drawing = True
        self.erasing = erase
        self.points = [pos]
        self._paint(pos)

    def continue_draw(self, pos):
        if not self.drawing:
            return
        if self.points:
            last = self.points[-1]
            self._draw_line(last, pos)
        self.points.append(pos)

    def stop_draw(self):
        self.drawing = False
        self.erasing = False
        self.points = []

    def _paint(self, pos):
        color = BASE if self.erasing else SURFACE0
        pygame.draw.circle(self.surface, color, pos, self.brush_size)

    def _draw_line(self, start, end):
        color = BASE if self.erasing else SURFACE0
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = max(1, int((dx**2 + dy**2)**0.5))
        for i in range(0, dist, max(1, self.brush_size // 4)):
            t = i / dist
            x = int(start[0] + dx * t)
            y = int(start[1] + dy * t)
            pygame.draw.circle(self.surface, color, (x, y), self.brush_size)
        pygame.draw.circle(self.surface, color, end, self.brush_size)

    def set_start(self, pos, angle=0):
        self.start_x, self.start_y = pos
        self.start_angle = angle

    def change_brush_size(self, delta):
        self.brush_size = max(5, min(100, self.brush_size + delta))

    def save(self, name):
        os.makedirs(TRACK_DIR, exist_ok=True)
        img_path = os.path.join(TRACK_DIR, f"{name}.png")
        pygame.image.save(self.surface, img_path)
        meta_path = os.path.join(TRACK_DIR, f"{name}.json")
        meta = {
            "start_x": self.start_x,
            "start_y": self.start_y,
            "start_angle": self.start_angle,
            "brush_size": self.brush_size,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def load(self, name):
        img_path = os.path.join(TRACK_DIR, f"{name}.png")
        meta_path = os.path.join(TRACK_DIR, f"{name}.json")
        if not os.path.exists(img_path):
            return False
        loaded = pygame.image.load(img_path).convert()
        self.surface.blit(loaded, (0, 0))
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.start_x = meta.get("start_x", self.width // 2)
            self.start_y = meta.get("start_y", self.height * 3 // 4)
            self.start_angle = meta.get("start_angle", 0)
            self.brush_size = meta.get("brush_size", 40)
        return True

    def smooth(self, passes=3):
        """Smooth out small jitters in drawn track edges via blur + re-threshold."""
        arr = pygame.surfarray.pixels3d(self.surface)
        threshold = (BASE[0] + SURFACE0[0]) // 2
        mask = (arr[:, :, 0] > threshold).astype(np.float32)

        # Repeated box blur to iron out hand-drawn wobbles
        for _ in range(passes):
            padded = np.pad(mask, ((1, 1), (1, 1)), mode='edge')
            mask = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
                + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
                + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0

        # Re-threshold back to crisp two-color surface
        drivable = mask > 0.5
        arr[drivable, 0] = SURFACE0[0]
        arr[drivable, 1] = SURFACE0[1]
        arr[drivable, 2] = SURFACE0[2]
        arr[~drivable, 0] = BASE[0]
        arr[~drivable, 1] = BASE[1]
        arr[~drivable, 2] = BASE[2]
        del arr  # release surfarray lock

    def generate_checkpoints(self, spacing=150):
        """BFS flood fill from start to build distance field and extract checkpoints."""
        arr = pygame.surfarray.pixels3d(self.surface)
        drivable = arr[:, :, 0] > 50  # white-ish = drivable
        del arr  # release lock

        h, w = drivable.shape[1], drivable.shape[0]  # surfarray is (x, y)
        dist_field = np.full((w, h), -1, dtype=np.int32)

        sx, sy = int(self.start_x), int(self.start_y)
        if not (0 <= sx < w and 0 <= sy < h and drivable[sx, sy]):
            self.checkpoints = []
            self.distance_field = dist_field
            return

        # BFS from start
        queue = deque()
        queue.append((sx, sy))
        dist_field[sx, sy] = 0
        farthest = (sx, sy)
        max_dist = 0

        while queue:
            x, y = queue.popleft()
            d = dist_field[x, y]
            if d > max_dist:
                max_dist = d
                farthest = (x, y)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and dist_field[nx, ny] == -1 and drivable[nx, ny]:
                    dist_field[nx, ny] = d + 1
                    queue.append((nx, ny))

        self.distance_field = dist_field

        # Trace spine: greedy walk from farthest back to start following steepest descent
        spine = []
        cx, cy = farthest
        while dist_field[cx, cy] > 0:
            spine.append((cx, cy))
            best = None
            best_d = dist_field[cx, cy]
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and dist_field[nx, ny] >= 0:
                    if dist_field[nx, ny] < best_d:
                        best_d = dist_field[nx, ny]
                        best = (nx, ny)
            if best is None:
                break
            cx, cy = best
        spine.append((sx, sy))
        spine.reverse()  # now goes from start to farthest

        # Sample checkpoints at regular BFS-distance intervals along the spine
        self.checkpoints = []
        next_dist = spacing
        for px, py in spine:
            if dist_field[px, py] >= next_dist:
                self.checkpoints.append((px, py))
                next_dist += spacing

    def draw_checkpoints(self, screen, next_idx):
        """Draw checkpoint markers on the track."""
        for i, (cx, cy) in enumerate(self.checkpoints):
            if i < next_idx:
                color = OVERLAY0  # passed
            elif i == next_idx:
                color = GREEN  # next target
            else:
                color = SURFACE2  # future
            pygame.draw.circle(screen, color, (cx, cy), 8)
            if i == next_idx:
                # Draw a larger ring around the active checkpoint
                pygame.draw.circle(screen, GREEN, (cx, cy), 16, 2)

    def get_field_distance(self, x, y):
        """Return BFS distance at a pixel position, or 0 if out of bounds / wall."""
        if self.distance_field is None:
            return 0
        ix, iy = int(x), int(y)
        w, h = self.distance_field.shape
        if 0 <= ix < w and 0 <= iy < h and self.distance_field[ix, iy] >= 0:
            return self.distance_field[ix, iy]
        return 0

    def clear(self):
        self.surface.fill(BASE)

    def get_track_list(self):
        os.makedirs(TRACK_DIR, exist_ok=True)
        files = [f[:-4] for f in os.listdir(TRACK_DIR) if f.endswith(".png")]
        return sorted(files)

    def draw_to_screen(self, screen):
        screen.blit(self.surface, (0, 0))

    def draw_start_marker(self, screen):
        import math
        x, y = self.start_x, self.start_y
        rad = math.radians(self.start_angle)
        end_x = x + math.cos(rad) * 25
        end_y = y + math.sin(rad) * 25

        # Render marker at 4x then smoothscale down for anti-aliasing
        SS = 4
        size = 70
        surf = pygame.Surface((size * SS, size * SS), pygame.SRCALPHA)
        c = size * SS // 2
        pygame.draw.circle(surf, RED, (c, c), 8 * SS)
        ex = c + math.cos(rad) * 25 * SS
        ey = c + math.sin(rad) * 25 * SS
        pygame.draw.line(surf, RED, (c, c), (int(ex), int(ey)), 3 * SS)
        arrow_len = 8 * SS
        for a_off in [math.pi * 0.8, -math.pi * 0.8]:
            ax = int(ex + math.cos(rad + a_off) * arrow_len)
            ay = int(ey + math.sin(rad + a_off) * arrow_len)
            pygame.draw.line(surf, RED, (int(ex), int(ey)), (ax, ay), 3 * SS)
        surf = pygame.transform.smoothscale(surf, (size, size))
        screen.blit(surf, (x - size // 2, y - size // 2))

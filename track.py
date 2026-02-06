import pygame
import json
import os
from colors import BASE, SURFACE0, LAVENDER, RED


TRACK_DIR = os.path.join(os.path.dirname(__file__), "tracks")


class Track:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # The track surface: Surface0 = drivable, Base = wall
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
        pygame.draw.circle(screen, RED, (x, y), 8)
        pygame.draw.line(screen, RED, (x, y), (int(end_x), int(end_y)), 3)
        # Arrow head
        arrow_len = 8
        arr_angle1 = rad + math.pi * 0.8
        arr_angle2 = rad - math.pi * 0.8
        a1x = int(end_x + math.cos(arr_angle1) * arrow_len)
        a1y = int(end_y + math.sin(arr_angle1) * arrow_len)
        a2x = int(end_x + math.cos(arr_angle2) * arrow_len)
        a2y = int(end_y + math.sin(arr_angle2) * arrow_len)
        pygame.draw.line(screen, RED, (int(end_x), int(end_y)), (a1x, a1y), 3)
        pygame.draw.line(screen, RED, (int(end_x), int(end_y)), (a2x, a2y), 3)

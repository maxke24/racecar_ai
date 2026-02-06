import sys
import os
import math

# Suppress ALSA errors on systems without audio (e.g. WSL)
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
from car import Car
from track import Track
from neural_network import Population
from colors import (
    BASE, MANTLE, CRUST, SURFACE0, SURFACE1, SURFACE2,
    LAVENDER, BLUE, TEAL, GREEN, YELLOW, PEACH, RED, MAUVE,
    TEXT, SUBTEXT0, OVERLAY0,
)

# ─── Constants ───────────────────────────────────────────────────────────────
# Internal (logical) resolution — all game logic runs at this size
INTERNAL_WIDTH, INTERNAL_HEIGHT = 1920, 1080
# Default window size
WIN_WIDTH, WIN_HEIGHT = 1280, 720
FPS = 60
POPULATION_SIZE = 50
MAX_STALL_FRAMES = 180

# ─── Modes ───────────────────────────────────────────────────────────────────
MODE_DRAW = "draw"
MODE_DRIVE = "drive"
MODE_AI = "ai"


class Game:
    def __init__(self):
        pygame.init()
        self.win_width = WIN_WIDTH
        self.win_height = WIN_HEIGHT
        self.screen = pygame.display.set_mode(
            (self.win_width, self.win_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Racecar AI")
        self.clock = pygame.time.Clock()

        # Internal surface for all game rendering (fixed resolution)
        self.game_surface = pygame.Surface((INTERNAL_WIDTH, INTERNAL_HEIGHT))
        self.aspect_ratio = INTERNAL_WIDTH / INTERNAL_HEIGHT

        # Letterbox offset for rendering
        self.render_x = 0
        self.render_y = 0
        self.render_w = self.win_width
        self.render_h = self.win_height
        self._update_letterbox()

        # Try nicer fonts, fall back to monospace
        for name in ["DejaVu Sans", "Segoe UI", "Ubuntu", "monospace"]:
            self.font = pygame.font.SysFont(name, 20, bold=False)
            if self.font:
                break
        for name in ["DejaVu Sans", "Segoe UI", "Ubuntu", "monospace"]:
            self.big_font = pygame.font.SysFont(name, 30, bold=True)
            if self.big_font:
                break

        self.track = Track(INTERNAL_WIDTH, INTERNAL_HEIGHT)
        self.mode = MODE_DRAW
        self.running = True

        # Manual driving car
        self.player_car = None

        # AI
        self.population = None
        self.ai_cars = []
        self.ai_generation_running = False
        self.ai_speed_multiplier = 1

        # Track saving
        self.track_name_input = ""
        self.typing_track_name = False
        self.show_track_list = False
        self.track_list = []

        # Start angle placement
        self.placing_start_angle = False
        self.start_place_pos = None

    def _update_letterbox(self):
        """Recalculate letterbox dimensions to maintain aspect ratio."""
        win_aspect = self.win_width / self.win_height
        if win_aspect > self.aspect_ratio:
            # Window is wider than content — pillarbox (bars on sides)
            self.render_h = self.win_height
            self.render_w = int(self.win_height * self.aspect_ratio)
            self.render_x = (self.win_width - self.render_w) // 2
            self.render_y = 0
        else:
            # Window is taller than content — letterbox (bars top/bottom)
            self.render_w = self.win_width
            self.render_h = int(self.win_width / self.aspect_ratio)
            self.render_x = 0
            self.render_y = (self.win_height - self.render_h) // 2

    def _map_mouse(self, pos):
        """Map window pixel coordinates to internal game coordinates."""
        x = int((pos[0] - self.render_x) * INTERNAL_WIDTH / self.render_w)
        y = int((pos[1] - self.render_y) * INTERNAL_HEIGHT / self.render_h)
        x = max(0, min(INTERNAL_WIDTH - 1, x))
        y = max(0, min(INTERNAL_HEIGHT - 1, y))
        return (x, y)

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    # ─── Events ──────────────────────────────────────────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if self.typing_track_name:
                self._handle_typing(event)
                continue

            if event.type == pygame.WINDOWSIZECHANGED:
                # Use WINDOWSIZECHANGED (SDL2) instead of VIDEORESIZE so the
                # event fires *after* the window manager finalises the size
                # (e.g. after a snap), avoiding set_mode fighting the WM.
                self.win_width = max(320, event.x)
                self.win_height = max(180, event.y)
                self.screen = pygame.display.set_mode(
                    (self.win_width, self.win_height), pygame.RESIZABLE
                )
                self._update_letterbox()
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.KEYUP:
                pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mousedown(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouseup(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mousemotion(event)
            elif event.type == pygame.MOUSEWHEEL:
                if self.mode == MODE_DRAW:
                    self.track.change_brush_size(event.y * 5)

    def _handle_typing(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_RETURN:
            if self.track_name_input:
                self.track.save(self.track_name_input)
            self.typing_track_name = False
            self.track_name_input = ""
        elif event.key == pygame.K_ESCAPE:
            self.typing_track_name = False
            self.track_name_input = ""
        elif event.key == pygame.K_BACKSPACE:
            self.track_name_input = self.track_name_input[:-1]
        else:
            if event.unicode and event.unicode.isprintable():
                self.track_name_input += event.unicode

    def _handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE or (event.key == pygame.K_q and not self.typing_track_name):
            self.running = False
            return

        if event.key == pygame.K_1:
            self.mode = MODE_DRAW
            self._stop_ai()
        elif event.key == pygame.K_2:
            self.mode = MODE_DRIVE
            self._stop_ai()
            self._spawn_player_car()
        elif event.key == pygame.K_3:
            self.mode = MODE_AI
            self._start_ai()

        if self.mode == MODE_DRAW:
            if event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.typing_track_name = True
                self.track_name_input = ""
            elif event.key == pygame.K_l and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.show_track_list = not self.show_track_list
                if self.show_track_list:
                    self.track_list = self.track.get_track_list()
            elif event.key == pygame.K_c:
                self.track.clear()
            elif event.key >= pygame.K_0 and event.key <= pygame.K_9 and self.show_track_list:
                idx = event.key - pygame.K_0
                if idx < len(self.track_list):
                    self.track.load(self.track_list[idx])
                    self.show_track_list = False

        if self.mode == MODE_DRIVE:
            if event.key == pygame.K_r:
                self._spawn_player_car()

        if self.mode == MODE_AI:
            if event.key == pygame.K_r:
                self._start_ai()
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.ai_speed_multiplier = min(10, self.ai_speed_multiplier + 1)
            elif event.key == pygame.K_MINUS:
                self.ai_speed_multiplier = max(1, self.ai_speed_multiplier - 1)
            elif event.key == pygame.K_b:
                self.population.save_best("tracks/best_brain.json")

    def _handle_mousedown(self, event):
        if self.mode == MODE_DRAW:
            pos = self._map_mouse(event.pos)
            if event.button == 1:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.placing_start_angle = True
                    self.start_place_pos = pos
                    self.track.set_start(pos)
                else:
                    self.track.start_draw(pos, erase=False)
            elif event.button == 3:
                self.track.start_draw(pos, erase=True)

    def _handle_mouseup(self, event):
        if self.mode == MODE_DRAW:
            pos = self._map_mouse(event.pos)
            if event.button == 1:
                if self.placing_start_angle and self.start_place_pos:
                    dx = pos[0] - self.start_place_pos[0]
                    dy = pos[1] - self.start_place_pos[1]
                    angle = math.degrees(math.atan2(dy, dx))
                    self.track.set_start(self.start_place_pos, angle)
                    self.placing_start_angle = False
                    self.start_place_pos = None
                else:
                    self.track.stop_draw()
            elif event.button == 3:
                self.track.stop_draw()

    def _handle_mousemotion(self, event):
        if self.mode == MODE_DRAW and not self.placing_start_angle:
            self.track.continue_draw(self._map_mouse(event.pos))

    # ─── Game logic ──────────────────────────────────────────────────────────
    def _spawn_player_car(self):
        self.player_car = Car(self.track.start_x, self.track.start_y, self.track.start_angle)

    def _start_ai(self):
        self.population = Population(POPULATION_SIZE)
        self.population.load_best("tracks/best_brain.json")
        self._spawn_ai_generation()

    def _spawn_ai_generation(self):
        self.ai_cars = []
        for i in range(POPULATION_SIZE):
            car = Car(self.track.start_x, self.track.start_y, self.track.start_angle)
            car.max_tire_marks = 0  # disable tire marks for AI perf
            self.ai_cars.append(car)
        self.ai_generation_running = True

    def _stop_ai(self):
        self.ai_generation_running = False
        self.ai_cars = []
        self.population = None

    def update(self):
        if self.mode == MODE_DRIVE and self.player_car and self.player_car.alive:
            keys = pygame.key.get_pressed()
            throttle = 0
            steering = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle = 1
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                throttle = -1
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steering = -1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steering = 1

            self.player_car.update(throttle, steering)
            self.player_car.cast_rays(self.track.surface)
            self.player_car.check_collision(self.track.surface)

        elif self.mode == MODE_AI and self.ai_generation_running:
            for _ in range(self.ai_speed_multiplier):
                self._update_ai()

    def _update_ai(self):
        if not self.ai_generation_running:
            return

        alive_count = 0
        for i, car in enumerate(self.ai_cars):
            if not car.alive:
                continue

            # Kill stalling cars: not moving for too long
            if car.stall_timer > MAX_STALL_FRAMES:
                car.alive = False
                continue

            car.cast_rays(self.track.surface)
            rays_norm = car.get_normalized_rays()
            speed_norm = car.speed / car.max_speed

            throttle, steering = self.population.get_actions(i, rays_norm, speed_norm)
            car.update(throttle, steering)
            car.check_collision(self.track.surface)

            if car.alive:
                alive_count += 1

        if alive_count == 0:
            fitnesses = [car.get_fitness() for car in self.ai_cars]
            self.population.evolve(fitnesses)
            self._spawn_ai_generation()

    # ─── Rendering ───────────────────────────────────────────────────────────
    def draw(self):
        self.game_surface.fill(CRUST)

        # Draw track
        self.track.draw_to_screen(self.game_surface)
        self.track.draw_start_marker(self.game_surface)

        if self.mode == MODE_DRAW:
            self._draw_draw_ui()
        elif self.mode == MODE_DRIVE:
            self._draw_drive_ui()
        elif self.mode == MODE_AI:
            self._draw_ai_ui()

        # Mode tabs at bottom
        self._draw_mode_tabs()

        # Scale game surface to window with letterboxing
        self.screen.fill((0, 0, 0))
        scaled = pygame.transform.smoothscale(self.game_surface, (self.render_w, self.render_h))
        self.screen.blit(scaled, (self.render_x, self.render_y))
        pygame.display.flip()

    def _draw_hud_panel(self, lines, x=10, y=10, color=TEXT):
        """Draw lines of text with a semi-transparent background panel."""
        if not lines:
            return
        line_h = 20
        pad = 8
        max_w = max(self.font.size(l)[0] for l in lines) + pad * 2
        panel_h = len(lines) * line_h + pad * 2

        panel = pygame.Surface((max_w, panel_h), pygame.SRCALPHA)
        panel.fill((*MANTLE, 200))
        self.game_surface.blit(panel, (x - pad, y - pad))

        for i, line in enumerate(lines):
            surface = self.font.render(line, True, color)
            self.game_surface.blit(surface, (x, y + i * line_h))

    def _draw_mode_tabs(self):
        modes = [("1: Draw", MODE_DRAW), ("2: Drive", MODE_DRIVE), ("3: AI", MODE_AI)]
        tab_w = 100
        tab_h = 28
        start_x = (INTERNAL_WIDTH - len(modes) * (tab_w + 8)) // 2
        y = INTERNAL_HEIGHT - tab_h - 6

        for i, (label, mode) in enumerate(modes):
            x = start_x + i * (tab_w + 8)
            active = self.mode == mode
            bg = LAVENDER if active else SURFACE1
            fg = CRUST if active else SUBTEXT0
            pygame.draw.rect(self.game_surface, bg, (x, y, tab_w, tab_h), border_radius=6)
            txt = self.font.render(label, True, fg)
            tx = x + (tab_w - txt.get_width()) // 2
            ty = y + (tab_h - txt.get_height()) // 2
            self.game_surface.blit(txt, (tx, ty))

    def _draw_draw_ui(self):
        # Brush cursor (map window mouse pos to internal coords)
        mx, my = self._map_mouse(pygame.mouse.get_pos())
        pygame.draw.circle(self.game_surface, LAVENDER, (mx, my), self.track.brush_size, 2)

        lines = [
            "DRAW MODE",
            f"Brush size: {self.track.brush_size} (scroll to change)",
            "Left click: paint track | Right click: erase",
            "Shift+click & drag: place start position & direction",
            "C: clear | Ctrl+S: save | Ctrl+L: load",
        ]
        self._draw_hud_panel(lines, color=LAVENDER)

        # Save dialog
        if self.typing_track_name:
            dw, dh = 300, 60
            dx, dy = INTERNAL_WIDTH // 2 - dw // 2, INTERNAL_HEIGHT // 2 - dh // 2
            pygame.draw.rect(self.game_surface, MANTLE, (dx, dy, dw, dh), border_radius=8)
            pygame.draw.rect(self.game_surface, LAVENDER, (dx, dy, dw, dh), 2, border_radius=8)
            lbl = self.font.render("Track name:", True, SUBTEXT0)
            self.game_surface.blit(lbl, (dx + 10, dy + 8))
            inp = self.font.render(self.track_name_input + "_", True, TEXT)
            self.game_surface.blit(inp, (dx + 10, dy + 32))

        # Track list
        if self.show_track_list:
            dw = 300
            dh = 30 + len(self.track_list) * 22
            dx = INTERNAL_WIDTH // 2 - dw // 2
            pygame.draw.rect(self.game_surface, MANTLE, (dx, 100, dw, dh), border_radius=8)
            pygame.draw.rect(self.game_surface, LAVENDER, (dx, 100, dw, dh), 2, border_radius=8)
            lbl = self.font.render("Load track (press number):", True, SUBTEXT0)
            self.game_surface.blit(lbl, (dx + 10, 105))
            for i, name in enumerate(self.track_list[:10]):
                txt = self.font.render(f"  {i}: {name}", True, TEXT)
                self.game_surface.blit(txt, (dx + 10, 125 + i * 22))

    def _draw_drive_ui(self):
        if self.player_car:
            self.player_car.draw(self.game_surface, color=LAVENDER, draw_rays=True)
            lines = [
                "DRIVE MODE",
                "Arrow keys / WASD: drive | R: reset",
                f"Speed: {self.player_car.speed:.1f}",
                f"Drift: {self.player_car.drift_amount:.2f}",
                f"Alive: {self.player_car.alive}",
            ]
            ray_str = " ".join(f"{d:.0f}" for d in self.player_car.ray_distances)
            lines.append(f"Rays: [{ray_str}]")
        else:
            lines = ["DRIVE MODE", "Press R to spawn car"]

        self._draw_hud_panel(lines, color=GREEN)

    def _draw_ai_ui(self):
        if not self.population:
            return

        alive = 0
        best_car = None
        best_fitness = -1
        for car in self.ai_cars:
            if car.alive:
                alive += 1
                f = car.get_fitness()
                if f > best_fitness:
                    best_fitness = f
                    best_car = car
                car.draw(self.game_surface, color=TEAL, draw_rays=False)

        if best_car:
            best_car.draw(self.game_surface, color=YELLOW, draw_rays=True)

        lines = [
            "AI MODE",
            f"Generation: {self.population.generation}",
            f"Alive: {alive}/{POPULATION_SIZE}",
            f"Best fitness (all time): {self.population.best_fitness:.0f}",
            f"Speed: {self.ai_speed_multiplier}x (+/- to change)",
            "R: restart | B: save best brain",
        ]
        self._draw_hud_panel(lines, color=PEACH)

    def _draw_text(self, text, x, y, color=TEXT):
        surface = self.font.render(text, True, color)
        self.game_surface.blit(surface, (x, y))


if __name__ == "__main__":
    game = Game()
    game.run()

import math
import pygame
import numpy as np
from colors import LAVENDER, TEAL, YELLOW, RED, OVERLAY0, CRUST, SURFACE2, MANTLE


class Car:
    def __init__(self, x, y, angle=0):
        self.start_x = x
        self.start_y = y
        self.start_angle = angle
        self.reset()

        # Physics constants
        self.max_speed = 8.0
        self.acceleration = 0.3
        self.brake_strength = 0.4
        self.friction = 0.05
        self.turn_speed = 4.5

        # Drift physics — Tokyo Drift style
        self.grip = 0.35           # how fast lateral velocity decays normally
        self.drift_grip = 0.08     # very low grip when drifting
        self.drift_threshold = 0.3 # easier to enter drift
        self.lateral_factor = 0.45 # how much steering pushes sideways
        self.drift_angle_factor = 0.15  # visual body rotation during drift

        # Raycast settings
        self.num_rays = 7
        self.ray_length = 200
        self.ray_angles = np.linspace(-90, 90, self.num_rays)
        self.ray_distances = np.ones(self.num_rays) * self.ray_length

        # Car shape
        self.length = 24
        self.width = 12

        # Tire marks
        self.tire_marks = []  # list of (x, y, alpha) for fading marks
        self.max_tire_marks = 600

        # Fitness tracking
        self.alive = True
        self.distance_driven = 0.0
        self.frames_alive = 0
        self.checkpoints_hit = 0
        self.stall_timer = 0
        self.prev_x = 0.0
        self.prev_y = 0.0

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.vel_forward = 0.0
        self.vel_lateral = 0.0
        self.drift_amount = 0.0
        self.visual_drift_angle = 0.0
        self.alive = True
        self.distance_driven = 0.0
        self.frames_alive = 0
        self.checkpoints_hit = 0
        self.stall_timer = 0
        self.prev_x = self.start_x
        self.prev_y = self.start_y
        self.ray_distances = np.ones(self.num_rays if hasattr(self, 'num_rays') else 7) * 200
        self.tire_marks = [] if hasattr(self, 'tire_marks') else []

    @property
    def speed(self):
        """Backward-compatible speed property."""
        return self.vel_forward

    def _get_wheel_positions(self):
        """Get the 4 wheel positions in world space."""
        rad = math.radians(self.angle + self.visual_drift_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        half_l = self.length / 2 - 2
        half_w = self.width / 2 + 1

        wheels = []
        for dx, dy in [(-half_l, -half_w), (-half_l, half_w),
                        (half_l, -half_w), (half_l, half_w)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            wheels.append((self.x + rx, self.y + ry))
        return wheels

    def get_corners(self):
        """Get the car body rectangle corners with drift visual rotation."""
        rad = math.radians(self.angle + self.visual_drift_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        half_l = self.length / 2
        half_w = self.width / 2

        corners = []
        for dx, dy in [(-half_l, -half_w), (half_l, -half_w),
                        (half_l, half_w), (-half_l, half_w)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners.append((self.x + rx, self.y + ry))
        return corners

    def get_collision_corners(self):
        """Get bounding corners for collision checks (uses actual angle, not visual)."""
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        half_l = self.length / 2 + 2
        half_w = self.width / 2 + 1

        corners = []
        for dx, dy in [(-half_l, -half_w), (half_l, -half_w),
                        (half_l, half_w), (-half_l, half_w)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners.append((self.x + rx, self.y + ry))
        return corners

    def update(self, throttle, steering):
        if not self.alive:
            return

        # Steering builds lateral velocity at speed
        if abs(self.vel_forward) > 0.1:
            speed_ratio = abs(self.vel_forward) / self.max_speed
            turn = steering * self.turn_speed * (self.vel_forward / self.max_speed)
            self.angle += turn
            # Steering at speed pushes laterally — the core drift mechanic
            self.vel_lateral += steering * abs(self.vel_forward) * self.lateral_factor

        # Acceleration / braking
        if throttle > 0:
            self.vel_forward += throttle * self.acceleration
        elif throttle < 0:
            self.vel_forward += throttle * self.brake_strength

        # Friction on forward velocity
        if self.vel_forward > 0:
            self.vel_forward = max(0, self.vel_forward - self.friction)
        elif self.vel_forward < 0:
            self.vel_forward = min(0, self.vel_forward + self.friction)

        # Clamp forward speed
        self.vel_forward = max(-self.max_speed * 0.3, min(self.max_speed, self.vel_forward))

        # Grip reduces lateral velocity — drift system
        self.drift_amount = abs(self.vel_lateral)
        if self.drift_amount > self.drift_threshold:
            current_grip = self.drift_grip
        else:
            current_grip = self.grip
        self.vel_lateral *= (1.0 - current_grip)

        # Visual drift angle — car body rotates into the drift
        target_drift_visual = -self.vel_lateral * self.drift_angle_factor * 60
        target_drift_visual = max(-35, min(35, target_drift_visual))
        self.visual_drift_angle += (target_drift_visual - self.visual_drift_angle) * 0.15

        # Tire marks when drifting hard
        if self.drift_amount > 0.8 and abs(self.vel_forward) > 1.0:
            wheels = self._get_wheel_positions()
            # Add rear wheel marks
            for w in wheels[:2]:  # rear wheels
                self.tire_marks.append((w[0], w[1], 180))
            if len(self.tire_marks) > self.max_tire_marks:
                self.tire_marks = self.tire_marks[-self.max_tire_marks:]

        # Move: forward direction + lateral (right) direction
        rad = math.radians(self.angle)
        fwd_x = math.cos(rad)
        fwd_y = math.sin(rad)
        right_x = -math.sin(rad)
        right_y = math.cos(rad)

        dx = fwd_x * self.vel_forward + right_x * self.vel_lateral
        dy = fwd_y * self.vel_forward + right_y * self.vel_lateral
        self.x += dx
        self.y += dy

        # Fitness: use euclidean displacement from start
        disp_x = self.x - self.start_x
        disp_y = self.y - self.start_y
        self.distance_driven = math.sqrt(disp_x * disp_x + disp_y * disp_y)
        self.frames_alive += 1

        # Track stalling: if car barely moved this frame, increment stall timer
        moved = math.sqrt((self.x - self.prev_x) ** 2 + (self.y - self.prev_y) ** 2)
        if moved < 0.5:
            self.stall_timer += 1
        else:
            self.stall_timer = max(0, self.stall_timer - 1)
        self.prev_x = self.x
        self.prev_y = self.y

    def cast_rays(self, track_surface):
        if not self.alive:
            return

        surface_width = track_surface.get_width()
        surface_height = track_surface.get_height()

        for i, ray_angle in enumerate(self.ray_angles):
            angle_rad = math.radians(self.angle + ray_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            dist = 0
            step = 3
            hit = False
            while dist < self.ray_length:
                dist += step
                px = int(self.x + cos_a * dist)
                py = int(self.y + sin_a * dist)

                if px < 0 or px >= surface_width or py < 0 or py >= surface_height:
                    hit = True
                    break

                pixel_color = track_surface.get_at((px, py))
                if pixel_color[0] < 50:  # Base color (~36) = wall
                    hit = True
                    break

            self.ray_distances[i] = dist if hit else self.ray_length

    def get_normalized_rays(self):
        return self.ray_distances / self.ray_length

    def check_collision(self, track_surface):
        if not self.alive:
            return

        corners = self.get_collision_corners()
        w = track_surface.get_width()
        h = track_surface.get_height()

        for cx, cy in corners:
            ix, iy = int(cx), int(cy)
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                self.alive = False
                return
            pixel = track_surface.get_at((ix, iy))
            if pixel[0] < 50:  # wall
                self.alive = False
                return

    def get_fitness(self):
        return self.distance_driven + self.checkpoints_hit * 100

    def draw(self, screen, color=TEAL, draw_rays=False):
        if not self.alive:
            return

        # Draw tire marks (fading skid marks)
        for i, (mx, my, alpha) in enumerate(self.tire_marks):
            if alpha > 10:
                a = min(255, alpha)
                mark_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
                mark_surf.fill((40, 40, 40, a))
                screen.blit(mark_surf, (int(mx) - 1, int(my) - 1))
                self.tire_marks[i] = (mx, my, alpha - 3)

        # Draw rays with gradient
        if draw_rays:
            for i, ray_angle in enumerate(self.ray_angles):
                angle_rad = math.radians(self.angle + ray_angle)
                end_x = self.x + math.cos(angle_rad) * self.ray_distances[i]
                end_y = self.y + math.sin(angle_rad) * self.ray_distances[i]

                ratio = max(0.0, min(1.0, self.ray_distances[i] / self.ray_length))
                # Lavender (far) to Red (close)
                ray_color = (
                    int(RED[0] + (LAVENDER[0] - RED[0]) * ratio),
                    int(RED[1] + (LAVENDER[1] - RED[1]) * ratio),
                    int(RED[2] + (LAVENDER[2] - RED[2]) * ratio),
                )
                pygame.draw.line(screen, ray_color,
                                 (int(self.x), int(self.y)),
                                 (int(end_x), int(end_y)), 2)

        # Visual angle includes drift rotation
        vis_rad = math.radians(self.angle + self.visual_drift_angle)
        cos_v = math.cos(vis_rad)
        sin_v = math.sin(vis_rad)

        def _rot(dx, dy):
            return (self.x + dx * cos_v - dy * sin_v,
                    self.y + dx * sin_v + dy * cos_v)

        hl = self.length / 2
        hw = self.width / 2

        # --- Car body (main rectangle) ---
        body = [_rot(-hl, -hw), _rot(hl, -hw), _rot(hl, hw), _rot(-hl, hw)]
        int_body = [(int(p[0]), int(p[1])) for p in body]
        pygame.draw.polygon(screen, color, int_body)
        pygame.draw.polygon(screen, OVERLAY0, int_body, 1)

        # --- Windshield (darker trapezoid on front half) ---
        ws_front = hl * 0.55
        ws_back = hl * 0.05
        ws_w = hw * 0.7
        windshield = [_rot(ws_back, -ws_w), _rot(ws_front, -ws_w),
                       _rot(ws_front, ws_w), _rot(ws_back, ws_w)]
        int_ws = [(int(p[0]), int(p[1])) for p in windshield]
        # Darken the body color for windshield
        ws_color = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
        pygame.draw.polygon(screen, ws_color, int_ws)

        # --- Headlights (two small bright rectangles at front) ---
        hl_size = 2
        for side in [-1, 1]:
            hx, hy = _rot(hl - 1, side * (hw - 2))
            pygame.draw.circle(screen, (255, 255, 200), (int(hx), int(hy)), hl_size)

        # --- Taillights (two small red rectangles at rear) ---
        for side in [-1, 1]:
            tx, ty = _rot(-hl + 1, side * (hw - 2))
            pygame.draw.circle(screen, RED, (int(tx), int(ty)), hl_size)

        # --- Wheels (4 dark rectangles) ---
        wheel_l = 5
        wheel_w = 2
        wheel_positions = [
            (-hl + 3, -hw - 1), (-hl + 3, hw + 1),  # rear
            (hl - 4, -hw - 1), (hl - 4, hw + 1),    # front
        ]
        for wx_off, wy_off in wheel_positions:
            wc = _rot(wx_off, wy_off)
            # Small rotated rectangle for wheel
            for ddx in range(-wheel_l // 2, wheel_l // 2 + 1):
                for ddy in range(-wheel_w // 2, wheel_w // 2 + 1):
                    px, py = _rot(wx_off + ddx * 0.8, wy_off + ddy * 0.8)
                    screen.set_at((int(px), int(py)), CRUST)

        # --- Drift smoke particles ---
        if self.drift_amount > 1.0 and abs(self.vel_forward) > 1.5:
            import random
            wheels = self._get_wheel_positions()
            for w in wheels[:2]:  # rear wheels smoke
                for _ in range(2):
                    sx = int(w[0] + random.randint(-4, 4))
                    sy = int(w[1] + random.randint(-4, 4))
                    size = random.randint(2, 5)
                    alpha = random.randint(40, 100)
                    smoke = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(smoke, (200, 200, 200, alpha), (size, size), size)
                    screen.blit(smoke, (sx - size, sy - size))

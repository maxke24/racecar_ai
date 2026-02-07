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

        # Drift physics — reduced drift for better AI training
        self.grip = 0.65           # how fast lateral velocity decays normally
        self.drift_grip = 0.25     # moderate grip when drifting
        self.drift_threshold = 0.6 # harder to enter drift state
        self.lateral_factor = 0.20 # less sideways push from steering
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
            self.vel_lateral -= steering * abs(self.vel_forward) * self.lateral_factor

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

        # Fitness: cumulative distance traveled
        moved = math.sqrt((self.x - self.prev_x) ** 2 + (self.y - self.prev_y) ** 2)
        self.distance_driven += moved
        self.frames_alive += 1

        # Track stalling: if car barely moved this frame, increment stall timer
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
                mark_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(mark_surf, (40, 40, 40, a), (2, 2), 2)
                screen.blit(mark_surf, (int(mx) - 2, int(my) - 2))
                self.tire_marks[i] = (mx, my, alpha - 3)

        # Draw rays with anti-aliased lines
        if draw_rays:
            for i, ray_angle in enumerate(self.ray_angles):
                angle_rad = math.radians(self.angle + ray_angle)
                end_x = self.x + math.cos(angle_rad) * self.ray_distances[i]
                end_y = self.y + math.sin(angle_rad) * self.ray_distances[i]

                ratio = max(0.0, min(1.0, self.ray_distances[i] / self.ray_length))
                ray_color = (
                    int(RED[0] + (LAVENDER[0] - RED[0]) * ratio),
                    int(RED[1] + (LAVENDER[1] - RED[1]) * ratio),
                    int(RED[2] + (LAVENDER[2] - RED[2]) * ratio),
                )
                pygame.draw.aaline(screen, ray_color,
                                   (self.x, self.y),
                                   (end_x, end_y))

        # --- Render car sprite with supersampling for smooth edges ---
        SS = 4  # supersample factor
        margin = 4
        sprite_w = (self.length + margin * 2) * SS
        sprite_h = (self.width + margin * 2 + 4) * SS  # extra for wheels
        sprite = pygame.Surface((sprite_w, sprite_h), pygame.SRCALPHA)

        cx = sprite_w / 2
        cy = sprite_h / 2
        hl = self.length / 2 * SS
        hw = self.width / 2 * SS

        # Wheels (drawn under the body)
        wl = 5 * SS
        ww = 2.5 * SS
        wheel_positions = [
            (-hl + 3 * SS, -hw - 1 * SS), (-hl + 3 * SS, hw + 1 * SS),
            (hl - 4 * SS, -hw - 1 * SS), (hl - 4 * SS, hw + 1 * SS),
        ]
        for wx_off, wy_off in wheel_positions:
            wheel = [
                (cx + wx_off - wl / 2, cy + wy_off - ww / 2),
                (cx + wx_off + wl / 2, cy + wy_off - ww / 2),
                (cx + wx_off + wl / 2, cy + wy_off + ww / 2),
                (cx + wx_off - wl / 2, cy + wy_off + ww / 2),
            ]
            pygame.draw.polygon(sprite, CRUST, wheel)

        # Car body
        body = [
            (cx - hl, cy - hw), (cx + hl, cy - hw),
            (cx + hl, cy + hw), (cx - hl, cy + hw),
        ]
        pygame.draw.polygon(sprite, color, body)
        pygame.draw.polygon(sprite, OVERLAY0, body, max(1, SS // 2))

        # Windshield
        ws_front = hl * 0.55
        ws_back = hl * 0.05
        ws_w = hw * 0.7
        windshield = [
            (cx + ws_back, cy - ws_w), (cx + ws_front, cy - ws_w),
            (cx + ws_front, cy + ws_w), (cx + ws_back, cy + ws_w),
        ]
        ws_color = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
        pygame.draw.polygon(sprite, ws_color, windshield)

        # Headlights
        hl_r = 2.5 * SS
        for side in [-1, 1]:
            pygame.draw.circle(sprite, (255, 255, 200),
                               (int(cx + hl - 1 * SS), int(cy + side * (hw - 2 * SS))),
                               int(hl_r))

        # Taillights
        for side in [-1, 1]:
            pygame.draw.circle(sprite, RED,
                               (int(cx - hl + 1 * SS), int(cy + side * (hw - 2 * SS))),
                               int(hl_r))

        # Downscale for anti-aliasing
        final_w = sprite_w // SS
        final_h = sprite_h // SS
        sprite = pygame.transform.smoothscale(sprite, (final_w, final_h))

        # Rotate and blit
        vis_angle = -(self.angle + self.visual_drift_angle)
        rotated = pygame.transform.rotozoom(sprite, vis_angle, 1.0)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect)

        # --- Drift smoke particles ---
        if self.drift_amount > 1.0 and abs(self.vel_forward) > 1.5:
            import random
            wheels = self._get_wheel_positions()
            for w in wheels[:2]:
                for _ in range(2):
                    sx = int(w[0] + random.randint(-4, 4))
                    sy = int(w[1] + random.randint(-4, 4))
                    size = random.randint(3, 6)
                    alpha = random.randint(40, 100)
                    smoke = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(smoke, (200, 200, 200, alpha), (size, size), size)
                    screen.blit(smoke, (sx - size, sy - size))

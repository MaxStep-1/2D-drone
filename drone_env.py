import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class DroneSlungLoadEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        wind_enabled=True,
        manual_wind=False,
        payload_mass=0.3,
        randomize_mass=False,
        mass_range=(0.1, 0.8),
        train_landing=False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.wind_enabled = wind_enabled
        self.manual_wind = manual_wind
        self.randomize_mass = randomize_mass
        self.mass_min, self.mass_max = mass_range
        self.train_landing = train_landing

        # Физические константы
        self.M = 1.0          # Масса дрона (кг)
        self.m = float(np.clip(payload_mass, 0.05, 1.5))  # Масса груза (кг)
        self.I = 0.08         # Момент инерции дрона (кг*м^2)
        self.arm_len = 0.22   # Полуразмах лучей дрона (м)
        self.cable_len = 1.0  # Длина троса (м)
        self.g = 9.81         # Ускорение свободного падения (м/с^2)

        # Сопротивление воздуха и демпфирование
        self.drag_drone = 0.25
        self.drag_payload = 0.15
        self.drag_rot = 0.02
        self.c_phi = 0.01

        self.dt = 0.02        # 50 Гц
        self.max_steps = 500

        # Границы мира (метры)
        self.x_limit = 6.0
        self.y_limit = 8.0

        # Параметры ветра
        self.wind_theta = 0.5
        self.wind_sigma = 1.5
        self.wind_mean = 0.0
        self.wind_x = 0.0

        # Режим посадки
        self.landing_mode = False
        self.landing_target_y = self.cable_len + 0.04  # Высота дрона при касании грузом земли

        # Нормализованная тяга левого и правого винтов [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Наблюдение (13 признаков):
        # [dx, dy, vx, vy, sin(theta), cos(theta), omega, sin(phi), cos(phi), phi_dot, wind_x, mass_norm, landing_flag]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Внутреннее состояние
        self.state = np.zeros(8, dtype=np.float64)
        self.target_x = 0.0
        self.target_y = 4.0
        self.last_thrusts = (0.0, 0.0)
        self.steps = 0

        # Визуализация
        self.screen = None
        self.clock = None
        self.font = None
        self.big_font = None
        self.width = 960
        self.height = 640
        self.scale = 80.0  # 80 пикселей = 1 метр

    @property
    def hover_thrust(self):
        #Текущая тяга с учётом массы дрона и груза
        return (self.M + self.m) * self.g

    def set_payload_mass(self, new_mass):
        #Задаёт новую массу груза в допустимых пределах
        self.m = float(np.clip(new_mass, 0.05, 1.5))

    def set_landing_mode(self, enabled, target_x=None):
        #Включает или выключает режим автоприземления
        self.landing_mode = bool(enabled)
        if self.landing_mode:
            if target_x is not None:
                self.target_x = float(np.clip(target_x, -self.x_limit + 1.0, self.x_limit - 1.0))
            self.target_y = self.landing_target_y

    def _derivatives(self, s, tl, tr, wind_x):
        #Правая часть системы ОДУ (уравнения Лагранжа 2-го рода)
        x, y, vx, vy, theta, omega, phi, phi_dot = s

        thrust = tl + tr
        torque = (tr - tl) * self.arm_len

        fx_thrust = -thrust * math.sin(theta)
        fy_thrust = thrust * math.cos(theta)
        q_dx = fx_thrust + self.drag_drone * (wind_x - vx)
        q_dy = fy_thrust - self.drag_drone * vy

        vpx = vx + self.cable_len * phi_dot * math.cos(phi)
        vpy = vy + self.cable_len * phi_dot * math.sin(phi)
        q_px = self.drag_payload * (wind_x - vpx)
        q_py = -self.drag_payload * vpy

        alpha = (torque - self.drag_rot * omega) / self.I

        cos_p = math.cos(phi)
        sin_p = math.sin(phi)
        total_m = self.M + self.m

        phi_ddot = (
            ((q_px / self.m - q_dx / self.M) * cos_p + (q_py / self.m - q_dy / self.M) * sin_p)
            / self.cable_len
            - (total_m / (self.m * self.M * self.cable_len ** 2)) * self.c_phi * phi_dot
        )

        ax = (
            q_dx + q_px
            + self.m * self.cable_len * (phi_dot ** 2) * sin_p
            - self.m * self.cable_len * phi_ddot * cos_p
        ) / total_m

        ay = (
            q_dy + q_py
            - self.m * self.cable_len * (phi_dot ** 2) * cos_p
            - self.m * self.cable_len * phi_ddot * sin_p
        ) / total_m - self.g

        return np.array([vx, vy, ax, ay, omega, alpha, phi_dot, phi_ddot], dtype=np.float64)

    def _rk4_step(self, tl, tr):
        #Интегрирование методом Рунге — Кутты 4-го порядка.
        dt = self.dt
        s = self.state
        w = self.wind_x

        k1 = self._derivatives(s, tl, tr, w)
        k2 = self._derivatives(s + 0.5 * dt * k1, tl, tr, w)
        k3 = self._derivatives(s + 0.5 * dt * k2, tl, tr, w)
        k4 = self._derivatives(s + dt * k3, tl, tr, w)

        self.state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.state[6] = (self.state[6] + math.pi) % (2.0 * math.pi) - math.pi

    def _get_obs(self):
        x, y, vx, vy, theta, omega, phi, phi_dot = self.state
        dx = (self.target_x - x) / 5.0
        dy = (self.target_y - y) / 5.0
        mass_norm = (self.m - 0.45) / 0.35
        return np.array([
            dx, dy,
            vx / 5.0, vy / 5.0,
            math.sin(theta), math.cos(theta), omega / 5.0,
            math.sin(phi), math.cos(phi), phi_dot / 5.0,
            self.wind_x / 5.0,
            mass_norm,
            1.0 if self.landing_mode else 0.0,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        if self.randomize_mass:
            self.m = float(rng.uniform(self.mass_min, self.mass_max))

        start_x = rng.uniform(-2.5, 2.5)
        start_y = rng.uniform(2.8, 5.5)
        init_theta = rng.uniform(-0.15, 0.15)
        init_phi = rng.uniform(-0.30, 0.30)

        self.state = np.array([start_x, start_y, 0.0, 0.0, init_theta, 0.0, init_phi, 0.0], dtype=np.float64)
        self.target_x = float(rng.uniform(-3.0, 3.0))

        # Во время обучения в 50% эпизодов тренируем автоприземление
        if self.train_landing and rng.random() < 0.5:
            self.landing_mode = True
            self.target_y = self.landing_target_y
        else:
            self.landing_mode = False
            self.target_y = float(rng.uniform(2.5, 6.0))

        if self.wind_enabled and not self.manual_wind:
            self.wind_mean = float(rng.uniform(-3.0, 3.0))
            self.wind_x = self.wind_mean
        else:
            self.wind_mean = 0.0
            self.wind_x = 0.0

        self.last_thrusts = (0.5 * self.hover_thrust, 0.5 * self.hover_thrust)
        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        tl = 0.5 * self.hover_thrust * (1.0 + action[0])
        tr = 0.5 * self.hover_thrust * (1.0 + action[1])
        self.last_thrusts = (tl, tr)

        if self.wind_enabled and not self.manual_wind:
            dw = (
                self.wind_theta * (self.wind_mean - self.wind_x) * self.dt
                + self.wind_sigma * math.sqrt(self.dt) * self.np_random.normal()
            )
            self.wind_x = float(np.clip(self.wind_x + dw, -6.0, 6.0))

        self._rk4_step(tl, tr)
        self.steps += 1

        x, y, vx, vy, theta, omega, phi, phi_dot = self.state
        payload_y = y - self.cable_len * math.cos(phi)

        landed = False
        crashed = False

        # Проверка мягкой посадки в режиме LANDING
        if self.landing_mode and payload_y <= 0.11:
            soft_touchdown = (
                abs(self.target_x - x) < 0.50
                and abs(vy) < 0.45
                and abs(vx) < 0.45
                and abs(theta) < 0.22
                and abs(phi) < 0.22
            )
            if soft_touchdown:
                landed = True
            elif payload_y <= 0.08:
                crashed = True

        # Условия аварии
        if not landed and (
            abs(x) > self.x_limit
            or y < 0.25
            or y > self.y_limit
            or (not self.landing_mode and payload_y < 0.08)
            or abs(theta) > math.pi / 2.0
        ):
            crashed = True

        # Расчёт награды
        dist = math.hypot(self.target_x - x, self.target_y - y)
        r_pos = 2.0 / (1.0 + dist)
        r_stab = (
            -0.35 * (theta ** 2)
            - 0.05 * (omega ** 2)
            - 0.70 * (phi ** 2)
            - 0.15 * (phi_dot ** 2)
            - 0.04 * (vx ** 2 + vy ** 2)
            - 0.02 * float(np.sum(action ** 2))
        )

        # В режиме посадки дополнительно поощряем гашение скорости у самой земли
        if self.landing_mode and payload_y < 1.0:
            r_stab -= 0.20 * (vx ** 2 + vy ** 2)

        r_bonus = 1.0 if (not self.landing_mode and dist < 0.35 and abs(phi) < 0.15) else 0.0
        reward = r_pos + r_stab + r_bonus

        if landed:
            reward += 100.0
        elif crashed:
            reward -= 50.0

        terminated = bool(crashed or landed)
        truncated = self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, {
            "dist": dist,
            "landed": landed,
            "crashed": crashed,
        }

    def _to_screen(self, wx, wy):
        sx = int((wx + self.x_limit) * self.scale)
        sy = int(self.height - wy * self.scale)
        return sx, sy

    def screen_to_world(self, sx, sy):
        wx = sx / self.scale - self.x_limit
        wy = (self.height - sy) / self.scale
        return float(np.clip(wx, -self.x_limit + 0.8, self.x_limit - 0.8)), float(np.clip(wy, 1.5, self.y_limit - 0.8))

    def render(self):
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("RL 2D Drone with Slung Load & Auto-Landing")
            else:
                self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 18)
            self.big_font = pygame.font.SysFont("consolas", 28, bold=True)

        # Фон и сетка
        self.screen.fill((20, 24, 34))
        for gx in range(-int(self.x_limit), int(self.x_limit) + 1):
            px, _ = self._to_screen(gx, 0)
            pygame.draw.line(self.screen, (32, 38, 52), (px, 0), (px, self.height), 1)
        for gy in range(0, int(self.y_limit) + 1):
            _, py = self._to_screen(0, gy)
            pygame.draw.line(self.screen, (32, 38, 52), (0, py), (self.width, py), 1)

        # Земля
        pygame.draw.rect(self.screen, (45, 58, 48), (0, self.height - 6, self.width, 6))

        # Посадочная площадка (если активен режим посадки)
        if self.landing_mode:
            pad_x, _ = self._to_screen(self.target_x, 0)
            pad_w = int(1.0 * self.scale)
            pygame.draw.rect(self.screen, (240, 200, 60), (pad_x - pad_w // 2, self.height - 10, pad_w, 10))

        # Целевая точка
        tx, ty = self._to_screen(self.target_x, self.target_y)
        target_color = (240, 200, 60) if self.landing_mode else (60, 220, 120)
        pygame.draw.circle(self.screen, target_color, (tx, ty), 16, 2)
        pygame.draw.line(self.screen, target_color, (tx - 22, ty), (tx + 22, ty), 1)
        pygame.draw.line(self.screen, target_color, (tx, ty - 22), (tx, ty + 22), 1)

        # Дрон, трос и груз
        x, y, _, _, theta, _, phi, _ = self.state
        dx_s, dy_s = self._to_screen(x, y)

        px_w = x + self.cable_len * math.sin(phi)
        py_w = y - self.cable_len * math.cos(phi)
        px_s, py_s = self._to_screen(px_w, py_w)

        payload_radius = int(np.clip(7 + 13 * self.m, 8, 22))
        pygame.draw.line(self.screen, (190, 200, 215), (dx_s, dy_s), (px_s, py_s), 2)
        pygame.draw.circle(self.screen, (235, 165, 50), (px_s, py_s), payload_radius)
        pygame.draw.circle(self.screen, (255, 220, 130), (px_s, py_s), payload_radius, 2)

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        lx, ly = self._to_screen(x - self.arm_len * cos_t, y - self.arm_len * sin_t)
        rx, ry = self._to_screen(x + self.arm_len * cos_t, y + self.arm_len * sin_t)

        tl, tr = self.last_thrusts
        for (ex, ey), thrust in (((lx, ly), tl), ((rx, ry), tr)):
            flame_len = (thrust / max(self.hover_thrust, 1e-6)) * 0.45
            if flame_len > 0.03:
                fx = ex + int(flame_len * math.sin(theta) * self.scale)
                fy = ey + int(flame_len * math.cos(theta) * self.scale)
                pygame.draw.line(self.screen, (80, 190, 255), (ex, ey), (fx, fy), 5)

        pygame.draw.line(self.screen, (230, 235, 245), (lx, ly), (rx, ry), 6)
        pygame.draw.circle(self.screen, (70, 150, 255), (dx_s, dy_s), 8)

        # Телеметрия
        mode_str = "LANDING" if self.landing_mode else "FLIGHT"
        telemetry = self.font.render(
            f"Mode: {mode_str} | Wind: {self.wind_x:+.2f} m/s | Mass: {self.m:.2f} kg | Phi: {math.degrees(phi):+5.1f} deg",
            True,
            (240, 210, 90) if self.landing_mode else (220, 225, 235),
        )
        self.screen.blit(telemetry, (16, 14))

        # Стрелка ветра
        cx, cy = self.width - 110, 32
        arrow_len = int(self.wind_x * 14)
        pygame.draw.line(self.screen, (110, 180, 255), (cx, cy), (cx + arrow_len, cy), 3)
        if abs(arrow_len) > 4:
            sign = 1 if arrow_len > 0 else -1
            tip = (cx + arrow_len, cy)
            pygame.draw.polygon(
                self.screen,
                (110, 180, 255),
                [tip, (tip[0] - 8 * sign, tip[1] - 5), (tip[0] - 8 * sign, tip[1] + 5)],
            )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
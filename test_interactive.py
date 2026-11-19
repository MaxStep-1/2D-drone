import os
import pygame
from stable_baselines3 import PPO
from drone_env import DroneSlungLoadEnv


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    model_path = os.path.join(models_dir, "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "ppo_drone_final.zip")

    if not os.path.exists(model_path):
        print(f"Модель не найдена в {models_dir}! Сначала запусти train.py.")
        return

    model = PPO.load(model_path)
    env = DroneSlungLoadEnv(
        render_mode="human",
        wind_enabled=True,
        manual_wind=True,
        payload_mass=0.3,
        randomize_mass=False,
        train_landing=False,
    )

    obs, _ = env.reset()
    running = True
    wind_fixed = False
    landed_pause = False

    print("Управление в окне:")
    print("  ЛКМ           — перенести цель в точку клика (отменяет посадку)")
    print("  L             — включить / выключить АВТОПРИЗЕМЛЕНИЕ")
    print("  Стрелки <- -> — изменить скорость ветра")
    print("  F             — зафиксировать / отпустить текущий ветер")
    print("  C             — обнулить ветер")
    print("  Q / -         — уменьшить массу груза")
    print("  E / +         — увеличить массу груза")
    print("  M             — сбросить массу груза к 0.30 кг")
    print("  R             — сбросить эпизод")
    print("  ESC           — выход")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    landed_pause = False
                elif event.key == pygame.K_l:
                    if not landed_pause:
                        env.set_landing_mode(not env.landing_mode)
                        print(f"Режим посадки: {'ВКЛ' if env.landing_mode else 'ВЫКЛ'}")
                elif event.key == pygame.K_f:
                    wind_fixed = not wind_fixed
                    print(f"Ветер {'зафиксирован' if wind_fixed else 'разблокирован'}: {env.wind_x:+.2f} m/s")
                elif event.key == pygame.K_c:
                    env.wind_x = 0.0
                    wind_fixed = False
                    print("Ветер обнулён")
                elif event.key in (pygame.K_e, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    env.set_payload_mass(env.m + 0.05)
                    print(f"Масса груза: {env.m:.2f} кг")
                elif event.key in (pygame.K_q, pygame.K_MINUS, pygame.K_KP_MINUS):
                    env.set_payload_mass(env.m - 0.05)
                    print(f"Масса груза: {env.m:.2f} кг")
                elif event.key == pygame.K_m:
                    env.set_payload_mass(0.30)
                    print("Масса груза сброшена к 0.30 кг")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                env.target_x, env.target_y = env.screen_to_world(mx, my)
                env.set_landing_mode(False)
                landed_pause = False

        keys = pygame.key.get_pressed()
        wind_changed = False
        if keys[pygame.K_RIGHT]:
            env.wind_x = min(env.wind_x + 0.08, 6.0)
            wind_changed = True
        elif keys[pygame.K_LEFT]:
            env.wind_x = max(env.wind_x - 0.08, -6.0)
            wind_changed = True

        if not wind_fixed and not wind_changed:
            env.wind_x *= 0.96
            if abs(env.wind_x) < 1e-3:
                env.wind_x = 0.0

        if not landed_pause:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if info.get("landed"):
                landed_pause = True
                print("Успешная мягкая посадка. Кликни в воздух для взлёта или нажми R.")
            elif terminated:
                obs, _ = env.reset()
            elif truncated:
                env.steps = 0
        else:
            # Отрисовка плашки об успешной посадке во время паузы
            env.render()
            if env.screen is not None and env.big_font is not None:
                banner = env.big_font.render(
                    "SOFT LANDING! Click in the air to take off",
                    True,
                    (90, 240, 140),
                )
                rect = banner.get_rect(center=(env.width // 2, env.height // 2 - 40))
                env.screen.blit(banner, rect)
                pygame.display.flip()

        if env.screen is not None:
            wind_label = "FIXED" if wind_fixed else "AUTO-DECAY"
            mode_label = "LANDED" if landed_pause else ("LANDING" if env.landing_mode else "FLIGHT")
            pygame.display.set_caption(
                f"RL 2D Drone [{mode_label}] | Mass: {env.m:.2f} kg | Wind: {env.wind_x:+.2f} m/s [{wind_label}]"
            )

    env.close()


if __name__ == "__main__":
    main()
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from drone_env import DroneSlungLoadEnv


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env_kwargs = {
        "wind_enabled": True,
        "randomize_mass": True,
        "mass_range": (0.1, 0.8),
        "train_landing": True,
    }

    print("Проверка среды Gymnasium...")
    check_env(DroneSlungLoadEnv(**env_kwargs), warn=True)
    print("Среда прошла проверку успешно!")

    train_env = make_vec_env(
        DroneSlungLoadEnv,
        n_envs=4,
        env_kwargs=env_kwargs,
    )

    eval_env = DroneSlungLoadEnv(**env_kwargs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.003,
        verbose=1,
        tensorboard_log=os.path.join(logs_dir, "tb"),
    )

    print("Старт обучения PPO (переменная масса + автоприземление)...")
    model.learn(total_timesteps=600_000, callback=eval_callback, progress_bar=False)

    final_path = os.path.join(models_dir, "ppo_drone_final")
    model.save(final_path)
    print(f"Обучение завершено! Лучшая модель: {os.path.join(models_dir, 'best_model.zip')}")


if __name__ == "__main__":
    main()
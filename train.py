# train.py
"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
- Trains a DQN agent and logs scores via ScoreLogger
- Optional evaluation with render
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

from agents.cartpole_dqn import DQNSolver, DQNConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_dqn.torch")


def train(num_episodes, terminal_penalty) -> DQNSolver:
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    print(f"[Info] Using device: {agent.device}")

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0  # small penalty to learn not to fail

            next_state = np.reshape(next_state, (1, obs_dim))
            agent.remember(state, action, reward, next_state, done)
            agent.experience_replay()
            state = next_state

            if done:
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()
    agent.save(MODEL_PATH)
    print(f"[Train] Model saved to {MODEL_PATH}")
    return agent


def evaluate(model_path, algorithm, episodes, render, fps):
    """
    Evaluate a trained RL agent.

    Args:
        model_path (str): Path to saved model file (.torch). If None, will look under 'models/' automatically.
        algorithm (str): Algorithm type, e.g. "dqn", "ppo", "a2c" — determines which agent class to use.
        episodes (int): Number of evaluation episodes.
        render (bool): Whether to show animation.
        fps (int): Render frame rate (default 60).
    """
    # -----------------------------
    # Step 1: Automatically locate model
    # -----------------------------
    model_dir = "models"
    if model_path is None:
        # Automatically find the first .torch file
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No saved model found in '{model_dir}/'. Please train first.")
        model_path = os.path.join(model_dir, candidates[0])
        print(f"[Eval] Using detected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    # -----------------------------
    # Step 2: Select appropriate agent type
    # -----------------------------
    
    if algorithm.lower() == "dqn":
        from agents.cartpole_dqn import DQNSolver, DQNConfig
        agent = DQNSolver(4, 2, cfg=DQNConfig())

    # -----------------------------
    # Step 3: Load model
    # -----------------------------
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    # -----------------------------
    # Step 4: Run evaluation
    # -----------------------------
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                q = agent.online(s_t)[0].cpu().numpy()
                action = int(np.argmax(q))

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1
            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores



if __name__ == "__main__":
    # Train first, then evaluate briefly.
    agent = train(num_episodes=10, terminal_penalty=True)
    evaluate(model_path="models/cartpole_dqn.torch", algorithm="dqn", episodes=5, render=True, fps=60)

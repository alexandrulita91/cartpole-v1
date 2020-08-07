# -*- coding: utf-8 -*-
import gym

import numpy as np
from brain import Brain


if __name__ == "__main__":
    recording_is_enabled = True

    env = gym.make('CartPole-v1')

    if recording_is_enabled:
        env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True)

    batch_size = 32
    num_episodes = 1000
    num_episode_steps = 500
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    brain = Brain(state_size=state_size, action_size=action_size)

    for episode in range(num_episodes):
        total_reward = 0
        observation = env.reset()
        current_state = np.reshape(observation, [1, state_size])

        for episode_step in range(num_episode_steps):
            env.render(mode="human")
            action = brain.act(current_state)

            observation, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            total_reward += reward

            next_state = np.reshape(observation, [1, state_size])
            brain.memorize(current_state, action, reward, next_state, done)
            current_state = next_state

            if done:
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode, num_episodes, episode_step, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode, num_episodes, episode_step, total_reward))

            if len(brain.memory) > batch_size:
                brain.replay(batch_size)

    env.close()

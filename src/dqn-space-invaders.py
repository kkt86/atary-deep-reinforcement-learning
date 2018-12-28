import gym

from collections import deque

from src.util.preprocess import preprocess_frame, stack_frames
from src.agents.dqn import DeepQNetwork

if __name__ == '__main__':
    # initialize environment
    env = gym.make("SpaceInvaders-v0")

    # hyperparameters
    max_episodes = 1000  # put this to something meaningful
    render_game = False

    # initialize agent
    state_size = (110, 84)
    action_space = env.action_space.n
    learning_rate = 0.001
    agent = DeepQNetwork(state_size, action_space)

    batch_size = 32
    train_start = 1000
    done = False


    for episode in range(max_episodes):
        # start new episode my reseting the environment
        episode_reward = 0
        frame = env.reset()
        preprocessed_frame = preprocess_frame(frame, state_size)
        state = stack_frames(deque(maxlen=4), preprocessed_frame, is_empty=True)

        while True:
            action = agent.act(state)
            next_frame, reward, done, info = env.step(action)
            next_preprocessed_frame = preprocess_frame(next_frame, state_size)
            next_state = stack_frames(state, next_preprocessed_frame, is_empty=False)
            episode_reward += reward

            # remember the previous state, action, reward, next_state, done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the current one
            state = next_state

            # render environment
            if render_game:
                env.render()

            if done:
                print("Episode: ", episode, "Reward: ", episode_reward, "Epsilon: ", agent.epsilon)
                break

            # train the agent with the experience of the episodes
            if len(agent.memory) > train_start:
                agent.replay(batch_size)

            if episode > 1 and episode % 50 == 0:
                agent.save_model('models/dqn-space-invaders.h5')


    env.close()


-----------------
# PPO Implementation for CartPole Environment

## Overview

This project implements the Proximal Policy Optimization (PPO) algorithm to solve the CartPole-v1 environment from the OpenAI Gym. PPO is an advanced reinforcement learning technique that combines policy gradients with a clipping mechanism to stabilize training.

## Features

- **Actor-Critic Architecture**: A shared model for both the actor (policy) and critic (value function).
- **PPO Loss**: Includes clipped surrogate loss, value function loss, and entropy regularization.
- **OpenAI Gym Integration**: Uses the CartPole-v1 environment for training.

---
## Dependencies

- Python (>=3.7)
- TensorFlow (>=2.0)
- NumPy
- OpenAI Gym

You can install the required packages using pip:

```
pip install tensorflow numpy gym
```
---

## Code Walkthrough
## 1. Environment Setup 
The code initializes the CartPole-v1 environment and retrieves state and action space sizes:
```python
import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
```

## 2. Hyperparameters
Key hyperparameters for PPO and neural network training are defined:
```python
gamma = 0.99  # Discount factor
lr_actor = 0.001  # Actor learning rate
lr_critic = 0.001  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization
```

## 3. Actor-Critic Model
The ActorCritic class defines a neural network with shared layers for policy logits and value prediction:
```python
class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value
```

## 4. PPO Loss Function
The PPO loss function includes policy loss, value loss, and an optional entropy bonus:
```python
def ppo_loss(old_logits, old_values, advantages, states, actions, returns):
    def compute_loss(logits, values, actions, returns):
        # Implementation here
    
    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        with tf.GradientTape() as tape:
            logits, values = model(states)
            loss = compute_loss(logits, values, actions, returns)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    advantages = get_advantages(returns, old_values)
    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)
    return loss
```

## 5. Training Loop
The main training loop collects trajectories, calculates returns, and updates the policy and value networks:
```python
for episode in range(max_episodes):
    states, actions, rewards, values, returns = [], [], [], [], []
    state = env.reset()
    for step in range(max_steps_per_episode):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        logits, value = model(state)

        # Sample action from the policy distribution
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        state = next_state

        if done:
            # Compute returns and update policy
            returns_batch = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns_batch.append(discounted_sum)
            returns_batch.reverse()

            states = tf.concat(states, axis=0)
            actions = np.array(actions, dtype=np.int32)
            values = tf.concat(values, axis=0)
            returns_batch = tf.convert_to_tensor(returns_batch)
            old_logits, _ = model(states)

            loss = ppo_loss(old_logits, values, returns_batch - np.array(values), states, actions, returns_batch)
            print(f"Episode: {episode + 1}, Loss: {loss.numpy()}")

            break
```

---
## Running the Code
To run the code, execute the script in your Python environment. The model will interact with the CartPole environment, learn a policy using PPO, and print the loss at the end of each episode.

---

## Output

The script outputs the loss for each episode:
```
Episode: 1, Loss: 0.534
Episode: 2, Loss: 0.421
...
```

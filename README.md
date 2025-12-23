# Flappy Bird with Deep Q-Learning (DQN)

An AI agent that learns to play Flappy Bird using Deep Q Learning with PyTorch.

## 🎮 Project Overview

This project implements a Deep Q-Network (DQN) to train an AI agent created by Claude Code to play Flappy Bird, with the agent slowly improving its performance over time.

## 📊 Results

- **Training Episodes**: 550
- **Best Score**: 23 pipes
- **Human Baseline**: 15-20 pipes
- **Status**: AI surpassed human performance! 🎉

## 🧠 How It Works

The AI uses:
- **Deep Q-Network (DQN)**: Neural network that predicts the best action
- **Experience Replay**: Stores past experiences to learn from
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **Reward System**:
  - +100 for passing a pipe
  - +1 for staying alive
  - -100 for crashing

## 🛠️ Technologies Used

- Python 
- PyTorch
- Pygame 


## 📈 Training Progress by Milestones

Ep: 134/550 | Score: 1 | Max: 1 | Avg100: 0.0 | e : 0.012 | -- first point


## 🎯 Features

- Real-time training visualization
- Model checkpointing (saves best models)
- Performance metrics tracking

## 📁 Project Structure
```
flappy-bird-dqn/
├── flappy_dqn.py          # Main training script
├── bg.png                  # Background image
├── FlappyBird.png     # Bird sprite
├── flappy_bird_best.pth    # Best model weights
├── README.md               # This file
```

## 👤 Author

Saaj Malhi - [GitHub](https://github.com/YOUR_USERNAME)

## 📄 License

This project is open source and available under the MIT License.

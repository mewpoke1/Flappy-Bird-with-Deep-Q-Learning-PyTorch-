# Flappy Bird with Deep Q-Learning (DQN)

An AI agent that learns to play Flappy Bird using Deep Q Learning with PyTorch.

## 🎮 Project Overview

This project implements a Deep Q-Network (DQN) to train an AI agent created by Claude Code to play Flappy Bird, with the agent slowly improving its performance over time.

## 📊 Results

- **Training Episodes**: 550 (can run up to 2000)
- **Best Score**: 23 points
- **Human Baseline**: 10-15 points
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

Ep: 134/550 | Score: 1 | Max: 1 | Avg100: 0.0 | e : 0.012 | -- AI got it's 1st point

Ep: 276/550 | Score: 2 | Max: 2 | Avg100: 0.1 | ε: 0.010 | -- AI got it's 2nd point

Ep: 334/550 | Score: 5 | Max: 5 | Avg100: 0.4 | ε: 0.010 | -- AI got it's 5th point

## 🎯 Features

- Real-time training visualization
- Model checkpointing (saves best models)
- Performance metrics tracking

## 📁 Project Structure
```
flappy-bird-dqn/
├── Pygame # normal folder
    ├── pygame2.py          # Main training script
├── bg.png                  # Background image
├── FlappyBird.png     # Bird sprite
```

## 👤 Author

Saaj Malhi - [GitHub](https://github.com/YOUR_USERNAME)

## 📄 License

This project is open source and available under the MIT License.

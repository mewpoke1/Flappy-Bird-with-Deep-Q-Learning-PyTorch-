import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time

pygame.init()
pygame.font.init()

# Vars
width, height = 1000, 800
plrwidth, plrheight = 70, 70

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.train_start = 1000
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.update_target_model()
        self.update_counter = 0
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_model()

# Game functions
def get_state(player, pipes, gravity_velocity):
    if len(pipes) == 0:
        return np.array([
            player.y / height,
            gravity_velocity / 20,
            1.0,
            0.5,
            0.5,
            1.0
        ])
    
    # Find the next pipe
    next_pipe = None
    for pipe in pipes:
        if pipe["top"].x + pipe["top"].width > player.x:
            next_pipe = pipe
            break
    
    if next_pipe is None:
        next_pipe = pipes[-1]
    
    # Distance to pipe
    dist_x = (next_pipe["top"].x - player.x) / width
    
    # Height of gaps
    top_gap = next_pipe["top"].height
    bottom_gap_start = next_pipe["top"].height + 250
    
    # Player position relative to gap
    gap_center = (top_gap + bottom_gap_start) / 2
    player_to_gap = (player.y - gap_center) / height
    
    state = np.array([
        player.y / height,
        gravity_velocity / 20,
        dist_x,
        top_gap / height,
        bottom_gap_start / height,
        player_to_gap
    ])
    
    return state

def draw(window, bg, plr, plrimg, et, pipes, score, episode, epsilon, Font, max_score):
    window.blit(bg, (0, 0))
    
    for pipe in pipes:
        pygame.draw.rect(window, (0, 200, 0), pipe["top"])
        pygame.draw.rect(window, (0, 200, 0), pipe["bottom"])
    
    time_text = Font.render(f"Time: {round(et)}s", 1, "red")
    window.blit(time_text, (10, 10))
    
    window.blit(plrimg, (plr.x, plr.y))
    
    score_text = Font.render(f"Score: {score}", 1, "red")
    window.blit(score_text, (10, 50))
    
    episode_text = Font.render(f"Episode: {episode}", 1, "red")
    window.blit(episode_text, (10, 90))
    
    epsilon_text = Font.render(f"Epsilon: {epsilon:.3f}", 1, "red")
    window.blit(epsilon_text, (10, 130))
    
    max_text = Font.render(f"Max Score: {max_score}", 1, "red")
    window.blit(max_text, (10, 170))
    
    pygame.display.update()

def main():
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Flappy Bird - DQN Training")
    
    bg = pygame.transform.scale(pygame.image.load("bg.png"), (width, height))
    plrimg = pygame.transform.scale(pygame.image.load("FlappyBird.png"), (plrwidth, plrheight))
    Font = pygame.font.SysFont("comicsans", 30)
    
    # Initialize agent
    state_size = 6
    action_size = 2  # 0: do nothing, 1: flap
    agent = DQNAgent(state_size, action_size)
    
    episodes = 2000
    max_score = 0
    scores = []
    
    for episode in range(episodes):
        gravity = 0.5
        gravity_velocity = 0
        player = pygame.Rect(200, height // 2, plrwidth, plrheight)
        
        start_time = time.time()
        elapsed_time = 0
        clock = pygame.time.Clock()
        
        pipes = []
        pipe_width = 70
        gap_height = 250
        pipe_speed = 3
        pipe_rate = 1500
        last_pipe = pygame.time.get_ticks() - pipe_rate + 500
        
        score = 0
        done = False
        steps = 0
        
        state = get_state(player, pipes, gravity_velocity)
        
        while not done:
            clock.tick(60)
            elapsed_time = time.time() - start_time
            steps += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    torch.save(agent.model.state_dict(), "flappy_bird_dqn.pth")
                    pygame.quit()
                    return
            
            # Agent decides action
            action = agent.act(state)
            
            # Apply action
            if action == 1:  # Flap
                gravity_velocity = -10
            
            # Physics
            gravity_velocity += gravity
            player.y += gravity_velocity
            
            # Calculate reward
            reward = 1  # Small reward for staying alive
            
            # Check boundaries
            if player.y > height - plrheight:
                done = True
                reward = -100
            elif player.y < 0:
                done = True
                reward = -100
            
            # Pipes
            current_time = pygame.time.get_ticks()
            if current_time - last_pipe > pipe_rate:
                pipe_height = random.randint(150, height - gap_height - 150)
                pipes.append({
                    "top": pygame.Rect(width, 0, pipe_width, pipe_height),
                    "bottom": pygame.Rect(width, pipe_height + gap_height, pipe_width, height),
                    "scored": False
                })
                last_pipe = current_time
            
            collision = False
            for pipe in pipes[:]:
                pipe["top"].x -= pipe_speed
                pipe["bottom"].x -= pipe_speed
                
                if not pipe["scored"] and pipe["top"].x + pipe_width < player.x:
                    score += 1
                    reward = 100  # Big reward for passing pipe
                    pipe["scored"] = True
                
                if pipe["top"].x < -pipe_width:
                    pipes.remove(pipe)
                
                if player.colliderect(pipe["top"]) or player.colliderect(pipe["bottom"]):
                    done = True
                    reward = -100
                    collision = True
            
            # Reward for being centered in the gap
            if not done and len(pipes) > 0:
                next_pipe = pipes[0]
                gap_center = (next_pipe["top"].height + next_pipe["top"].height + gap_height) / 2
                distance_from_center = abs(player.y + plrheight/2 - gap_center)
                if distance_from_center < gap_height / 3:
                    reward += 0.5
            
            next_state = get_state(player, pipes, gravity_velocity)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Train every step
            agent.replay()
            
            # Draw every 3 frames for speed
            if steps % 3 == 0:
                draw(window, bg, player, plrimg, elapsed_time, pipes, score, episode + 1, agent.epsilon, Font, max_score)
            
            if done:
                if score > max_score:
                    max_score = score
                    torch.save(agent.model.state_dict(), "flappy_bird_best.pth")
                
                scores.append(score)
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Ep: {episode + 1}/{episodes} | Score: {score} | Max: {max_score} | Avg100: {avg_score:.1f} | ε: {agent.epsilon:.3f} | Mem: {len(agent.memory)}")
                break
        
        # Save periodically
        if (episode + 1) % 100 == 0:
            torch.save(agent.model.state_dict(), f"flappy_bird_ep{episode+1}.pth")
    
    torch.save(agent.model.state_dict(), "flappy_bird_dqn_final.pth")
    print("Training complete!")
    pygame.quit()

if __name__ == "__main__":
    main()
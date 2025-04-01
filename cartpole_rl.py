import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import pickle
import os

class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        """
        Initialize Q-learning agent with parameters
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table with small random values
        self.q_table = {}
    
    def get_discrete_state(self, state):
        """
        Convert continuous state to discrete state for Q-table indexing
        """
        # Define state boundaries for CartPole
        cart_position_bins = np.linspace(-2.4, 2.4, 10)
        cart_velocity_bins = np.linspace(-4, 4, 10)
        pole_angle_bins = np.linspace(-0.418, 0.418, 10)
        pole_velocity_bins = np.linspace(-4, 4, 10)
        
        # Digitize the state
        discrete_state = (
            np.digitize(state[0], cart_position_bins),
            np.digitize(state[1], cart_velocity_bins),
            np.digitize(state[2], pole_angle_bins),
            np.digitize(state[3], pole_velocity_bins)
        )
        
        return discrete_state
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy
        """
        discrete_state = self.get_discrete_state(state)
        
        # Explore: choose random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)
        
        # Exploit: choose best action from Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space)
            
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table based on action and reward
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)
        
        # Initialize Q-values if states are not in the Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space)
        
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_space)
        
        # Q-learning formula
        current_q = self.q_table[discrete_state][action]
        
        # If episode is done, there is no next Q value
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
        
        # Compute new Q value using the Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[discrete_state][action] = new_q
        
        # Decay exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def save_model(self, filename):
        """
        Save the Q-table and agent parameters to a file
        """
        model_data = {
            'q_table': self.q_table,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filename):
        """
        Load the Q-table and agent parameters from a file
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        self.q_table = model_data['q_table']
        self.exploration_rate = model_data['exploration_rate']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']

def train_agent(num_episodes=1000, render_interval=100, save_visualization=False):
    """
    Train the Q-learning agent on CartPole
    """
    # Use 'rgb_array' render mode if we'll be visualizing training
    render_mode = 'rgb_array' if save_visualization else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    agent = QLearningAgent(
        state_space=env.observation_space.shape[0], 
        action_space=env.action_space.n
    )
    
    # Tracking metrics
    episode_rewards = []
    average_rewards = []
    
    # Store selected episode frames for visualization
    visualization_frames = {}
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        episode_frames = []
        
        while not (done or truncated):
            # Select action
            action = agent.choose_action(state)
            
            # Capture frame if visualization is enabled and we're at a visualization interval
            if save_visualization and episode % render_interval == 0:
                episode_frames.append(env.render())
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Modify reward to encourage learning
            if done or truncated:
                reward = -10  # Penalty for failing
            
            # Agent learns from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
        
        # Save frames for visualization
        if save_visualization and episode % render_interval == 0:
            save_animation(episode_frames, f'visualizations/training_episode_{episode}.gif')
        
        # Track episode rewards
        episode_rewards.append(total_reward)
        
        # Calculate running average of rewards
        average_reward = np.mean(episode_rewards[-100:])
        average_rewards.append(average_reward)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode: {episode}, Average Reward: {average_reward:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    agent.save_model('models/cartpole_q_learning.pkl')
    
    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(average_rewards, label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.close()
    
    env.close()
    return agent

def save_animation(frames, filename):
    """
    Save frames as an animation using matplotlib (as GIF)
    """
    # Convert filename to use .gif extension
    filename = filename.replace('.mp4', '.gif')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display first frame
    img = ax.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        img.set_data(frames[i])
        return [img]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=50, blit=True
    )
    
    # Save animation as GIF
    anim.save(filename, writer='pillow', fps=30)
    plt.close()
    print(f"Animation saved as '{filename}'")

def visualize_agent(agent, num_episodes=5, save_video=True, video_filename='cartpole_visualization.gif'):
    """
    Visualize the trained agent's performance with matplotlib
    """
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    all_frames = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_frames = []
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Capture frame
            episode_frames.append(env.render())
            
            # Get action from agent (pure exploitation)
            discrete_state = agent.get_discrete_state(state)
            
            if discrete_state in agent.q_table:
                action = np.argmax(agent.q_table[discrete_state])
            else:
                action = env.action_space.sample()
            
            # Take action
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        all_frames.extend(episode_frames)
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    
    env.close()
    
    # Save all frames as a continuous video
    if save_video and all_frames:
        save_animation(all_frames, video_filename)
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward

def visualize_with_state_info(agent, num_episodes=1, video_filename='cartpole_with_state_info.gif'):
    """
    Create a more detailed visualization showing state information
    """
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    frames = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Get raw frame
            raw_frame = env.render()
            
            # Get Q-values for current state
            discrete_state = agent.get_discrete_state(state)
            q_values = agent.q_table.get(discrete_state, np.zeros(2))
            
            # Get action
            action = np.argmax(q_values)
            
            # Create a figure with two subplots: game and state info
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot game frame
            ax1.imshow(raw_frame)
            ax1.set_title('CartPole Environment')
            ax1.axis('off')
            
            # Plot state info
            ax2.axis('off')
            ax2.set_title('State Information')
            state_text = f"Step: {step}\n\n"
            state_text += f"Cart Position: {state[0]:.4f}\n"
            state_text += f"Cart Velocity: {state[1]:.4f}\n"
            state_text += f"Pole Angle: {state[2]:.4f}\n"
            state_text += f"Pole Velocity: {state[3]:.4f}\n\n"
            state_text += f"Discrete State: {discrete_state}\n\n"
            state_text += f"Q-values:\n"
            state_text += f"  Left (0): {q_values[0]:.4f}\n"
            state_text += f"  Right (1): {q_values[1]:.4f}\n\n"
            state_text += f"Action: {'Left' if action == 0 else 'Right'}"
            
            # Add text to the right subplot
            ax2.text(0.1, 0.5, state_text, fontsize=12, verticalalignment='center')
            
            # Capture the figure as an image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)
            plt.close(fig)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            step += 1
    
    env.close()
    
    # Save the frames as a video
    save_animation(frames, video_filename)
    print(f"Detailed visualization saved as '{video_filename}'")

def compare_training_stages(agent_file='models/cartpole_q_learning.pkl', num_episodes=4):
    """
    Compare the agent's performance at different stages of training
    """
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # Create a new agent for comparison
    untrained_agent = QLearningAgent(
        state_space=env.observation_space.shape[0], 
        action_space=env.action_space.n
    )
    
    # Create a partially trained agent
    partially_trained_agent = QLearningAgent(
        state_space=env.observation_space.shape[0], 
        action_space=env.action_space.n
    )
    partially_trained_agent.exploration_rate = 0.5
    
    # Train the partially trained agent for a few episodes
    state, _ = env.reset()
    for _ in range(100):  # Train for 100 episodes
        done = False
        truncated = False
        while not (done or truncated):
            action = partially_trained_agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            partially_trained_agent.learn(state, action, reward, next_state, done)
            state = next_state
        state, _ = env.reset()
    
    # Load the fully trained agent
    fully_trained_agent = QLearningAgent(
        state_space=env.observation_space.shape[0], 
        action_space=env.action_space.n
    )
    fully_trained_agent.load_model(agent_file)
    
    # Evaluate each agent
    agents = [
        ("Untrained", untrained_agent),
        ("Partially Trained", partially_trained_agent),
        ("Fully Trained", fully_trained_agent)
    ]
    
    comparison_frames = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}:")
        episode_frames = []
        
        for agent_name, agent in agents:
            state, _ = env.reset()
            agent_frames = []
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Capture frame
                frame = env.render()
                agent_frames.append(frame)
                
                # Get action
                discrete_state = agent.get_discrete_state(state)
                if discrete_state in agent.q_table:
                    action = np.argmax(agent.q_table[discrete_state])
                else:
                    action = env.action_space.sample()
                
                # Take action
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            
            print(f"{agent_name}: Total Reward = {total_reward}")
            
            # If this is the first frame for this episode, initialize the comparison frame
            if not episode_frames:
                # Create a figure with subplots for each agent
                fig, axes = plt.subplots(1, len(agents), figsize=(15, 5))
                for i, (name, _) in enumerate(agents):
                    axes[i].set_title(f"{name}")
                    axes[i].axis('off')
                
                # Initialize the subplot images
                subplot_images = [ax.imshow(np.zeros_like(agent_frames[0])) for ax in axes]
                
                # Add episode number
                fig.suptitle(f"Episode {episode+1}", fontsize=16)
                
                # Capture the initial frame
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                episode_frames.append(img)
                plt.close(fig)
            
            # Add the frames to the episode frames
            max_length = max(len(agent_frames) for agent_name, agent in agents)
            for frame_idx in range(max_length):
                # Create a figure with subplots for each agent
                fig, axes = plt.subplots(1, len(agents), figsize=(15, 5))
                
                for i, (name, _) in enumerate(agents):
                    axes[i].set_title(f"{name}")
                    axes[i].axis('off')
                    
                    # Use the last frame if we've run out of frames for this agent
                    if frame_idx < len(agent_frames):
                        frame = agent_frames[frame_idx]
                    else:
                        frame = agent_frames[-1]
                    
                    axes[i].imshow(frame)
                
                # Add episode number
                fig.suptitle(f"Episode {episode+1}", fontsize=16)
                
                # Capture the figure as an image
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                episode_frames.append(img)
                plt.close(fig)
            
            comparison_frames.extend(episode_frames)
    
    env.close()
    
    # Save the frames as a video
    save_animation(comparison_frames, 'agent_comparison.gif')
    print(f"Agent comparison saved as 'agent_comparison.gif'")

def analyze_q_values(agent):
    """
    Analyze and visualize the Q-values of the trained agent
    """
    # Extract Q-values from the agent's Q-table
    q_values_list = []
    for state, q_values in agent.q_table.items():
        cart_pos_bin, cart_vel_bin, pole_angle_bin, pole_vel_bin = state
        q_values_list.append({
            'cart_pos_bin': cart_pos_bin,
            'cart_vel_bin': cart_vel_bin,
            'pole_angle_bin': pole_angle_bin,
            'pole_vel_bin': pole_vel_bin,
            'q_left': q_values[0],
            'q_right': q_values[1],
            'best_action': np.argmax(q_values)
        })
    
    # Create a figure for visualizing Q-values
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cart Position vs Pole Angle (averaging over velocity dimensions)
    position_angle_data = np.zeros((10, 10, 2))  # 10 bins for each dimension, 2 actions
    position_angle_count = np.zeros((10, 10, 2))
    
    for entry in q_values_list:
        pos_bin = entry['cart_pos_bin'] - 1  # Adjust for zero-indexing
        angle_bin = entry['pole_angle_bin'] - 1
        position_angle_data[pos_bin, angle_bin, 0] += entry['q_left']
        position_angle_data[pos_bin, angle_bin, 1] += entry['q_right']
        position_angle_count[pos_bin, angle_bin, 0] += 1
        position_angle_count[pos_bin, angle_bin, 1] += 1
    
    # Avoid division by zero
    position_angle_count[position_angle_count == 0] = 1
    position_angle_avg = position_angle_data / position_angle_count
    
    # Plot the best action for each state
    best_actions = np.argmax(position_angle_avg, axis=2)
    im = axes[0, 0].imshow(best_actions, cmap='coolwarm')
    axes[0, 0].set_title('Best Action (Position vs Angle)')
    axes[0, 0].set_xlabel('Pole Angle Bin')
    axes[0, 0].set_ylabel('Cart Position Bin')
    cbar = fig.colorbar(im, ax=axes[0, 0])
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Left', 'Right'])
    
    # Plot 2: Cart Velocity vs Pole Velocity (averaging over position dimensions)
    vel_data = np.zeros((10, 10, 2))
    vel_count = np.zeros((10, 10, 2))
    
    for entry in q_values_list:
        cart_vel_bin = entry['cart_vel_bin'] - 1
        pole_vel_bin = entry['pole_vel_bin'] - 1
        vel_data[cart_vel_bin, pole_vel_bin, 0] += entry['q_left']
        vel_data[cart_vel_bin, pole_vel_bin, 1] += entry['q_right']
        vel_count[cart_vel_bin, pole_vel_bin, 0] += 1
        vel_count[cart_vel_bin, pole_vel_bin, 1] += 1
    
    # Avoid division by zero
    vel_count[vel_count == 0] = 1
    vel_avg = vel_data / vel_count
    
    # Plot the best action for each state
    best_vel_actions = np.argmax(vel_avg, axis=2)
    im = axes[0, 1].imshow(best_vel_actions, cmap='coolwarm')
    axes[0, 1].set_title('Best Action (Cart Velocity vs Pole Velocity)')
    axes[0, 1].set_xlabel('Pole Velocity Bin')
    axes[0, 1].set_ylabel('Cart Velocity Bin')
    cbar = fig.colorbar(im, ax=axes[0, 1])
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Left', 'Right'])
    
    # Plot 3: Heatmap of Q-value differences
    q_diff = np.abs(position_angle_avg[:, :, 0] - position_angle_avg[:, :, 1])
    im = axes[1, 0].imshow(q_diff, cmap='viridis')
    axes[1, 0].set_title('Q-value Difference (Position vs Angle)')
    axes[1, 0].set_xlabel('Pole Angle Bin')
    axes[1, 0].set_ylabel('Cart Position Bin')
    fig.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Heatmap of Q-value differences for velocities
    q_vel_diff = np.abs(vel_avg[:, :, 0] - vel_avg[:, :, 1])
    im = axes[1, 1].imshow(q_vel_diff, cmap='viridis')
    axes[1, 1].set_title('Q-value Difference (Cart Velocity vs Pole Velocity)')
    axes[1, 1].set_xlabel('Pole Velocity Bin')
    axes[1, 1].set_ylabel('Cart Velocity Bin')
    fig.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('q_value_analysis.png')
    plt.close()
    print("Q-value analysis saved as 'q_value_analysis.png'")

def main():
    """
    Main function to train and evaluate the agent
    """
    # Check if trained model exists
    model_path = 'models/cartpole_q_learning.pkl'
    train_new_model = True
    
    if os.path.exists(model_path):
        response = input("A trained model already exists. Train a new one? (y/n): ")
        train_new_model = response.lower() == 'y'
    
    if train_new_model:
        print("Starting Q-learning agent training for CartPole...")
        agent = train_agent(num_episodes=1000, save_visualization=True, render_interval=200)
        print("\nTraining completed.")
    else:
        # Load existing model
        print("Loading trained model...")
        env = gym.make('CartPole-v1')
        agent = QLearningAgent(
            state_space=env.observation_space.shape[0], 
            action_space=env.action_space.n
        )
        agent.load_model(model_path)
        env.close()
    
    # Visualization options
    print("\nVisualization options:")
    print("1. Standard visualization")
    print("2. Detailed visualization with state information")
    print("3. Compare training stages")
    print("4. Analyze Q-values")
    print("5. All visualizations")
    print("6. Skip visualization")
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == '1':
        visualize_agent(agent, num_episodes=3)
    elif choice == '2':
        visualize_with_state_info(agent)
    elif choice == '3':
        compare_training_stages(agent_file=model_path)
    elif choice == '4':
        analyze_q_values(agent)
    elif choice == '5':
        print("\nGenerating all visualizations...")
        visualize_agent(agent, num_episodes=2)
        visualize_with_state_info(agent)
        compare_training_stages(agent_file=model_path)
        analyze_q_values(agent)
    else:
        print("Skipping visualization.")
    
    print("\nProgram completed.")

if __name__ == "__main__":
    main()
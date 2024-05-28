import time
from warehouse import Warehouse  # Assuming the class name and import path are correct

def main():
    # Initialize your environment with the correct parameters
    env = Warehouse(
        shelf_columns=9, 
        column_height=8, 
        shelf_rows=3, 
        n_agents=10, 
        msg_bits=3, 
        sensor_range=1, 
        request_queue_size=5, 
        reward_type='GLOBAL',
        max_inactivity_steps=1000,  # Example value, adjust as necessary
        max_steps=5000              # Example value, adjust as necessary
    )

    # Reset the environment
    observation = env.reset()

    start_time = time.time()  # Start timing

    # Run for 1000 timesteps
    for _ in range(1000):
        action = env.action_space.sample()  # Assuming this method of choosing actions is acceptable
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()  # Reset if the environment reaches an endpoint

    end_time = time.time()  # End timing

    # Calculate and print the duration and timesteps per second
    duration = end_time - start_time
    timesteps_per_second = 1000 / duration
    print(f"Processed 1000 timesteps in {duration:.2f} seconds.")
    print(f"Timesteps per second: {timesteps_per_second:.2f}")

if __name__ == "__main__":
    main()

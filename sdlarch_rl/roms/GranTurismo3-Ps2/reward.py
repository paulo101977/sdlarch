def reward(previous_state: dict, state: dict):
    """
    Calculate the reward based on the current and next state.
    
    Args:
        previous_state (dict): The oldest state of the environment.
        state (dict): The current state of the environment.
    
    Returns:
        tuple: A tuple containing the reward and a boolean indicating if the episode is done.
    """
    reward = 0.0
    done = False

    # velocity, position, current_lap, max_lap, wrong_way

    current_position = state['position']
    previous_position = previous_state['position']

    # car start in 6th position
    # The car has moved forward
    if current_position > 0:
        reward += 1.0/6 * (6 - current_position)  # Reward for moving up in position
    
    if current_position < previous_position:
        reward += 0.3  # Reward for moving forward in position

    if state['wrong_way']:
        reward -= 1.0  # Penalty for going the wrong way
    else:
        if state['velocity'] > 1 and state['velocity'] < 10:
            reward += 1 / 200 # Reward for maintaining speed
        elif state['velocity'] > 10:
            reward += 1 / 200 * state['velocity']  # Reward for maintaining speed
        else:
            reward -= 0.5   # Penalty for low speed

    if state['current_lap'] > 0 and state['max_lap'] > 0 and state['current_lap'] > state['max_lap'] \
        and state['current_lap'] < 5 and state['max_lap'] < 5 :
        done = True

    # Ensure the reward is within the range [-1.0, 1.0]
    if reward < -1.0:
        reward = -1.0
    if reward > 1.0:
        reward = 1.0

    return reward, done
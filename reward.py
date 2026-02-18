reward_values = {
    # If a level state is loaded, but Link is in the overworld
    'level_state_in_overworld': -2,
    # If Link goes into a new room (reduced to prevent doorway exploitation)
    'new_room': -100000,
    # If Link moves around the room
    'movement': 5,
    # If Link has already been to this state (penalty for getting stuck)
    'repeat_state': -0.05,
    # Reward for killing an enemy (PRIMARY goal - make this dominant)
    'kill_enemy': 100
}
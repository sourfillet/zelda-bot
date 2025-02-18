from reward import reward_values

dungeon_save_states = set([
    # a list of the dungeon save state names
    "level1",
    "level2",
    "level3",
    "level4",
    "level5",
    "level6",
    "level7",
    "level8"
])

single_pickup_items = [
    # these can only be picked up once, and won't decrease
    "Boomerang",
    "Bow",
    "Candle",
    "Flute",
    "Ladder",
    "Letter",
    "Magic Book",
    "Magical Key",
    "Magical Rod",
    "Power Bracelet",
    "Raft",
    "Ring",
    "Shield",
    "Sword"
]

multi_pickup_items = [
    # these can be picked up multiple times, and will decrease
    "Arrow",
    "Bombs",
    "Keys",
    "Rupees"
]

def get_actual_hearts(hearts=0):
    """
        Get the actual number of hearts based on the value of the hearts variable.
    """
    partials = {
                256:4,
                128:3.5,
                255:3,
                127:2.5,
                254:2,
                126:1.5,
                253:1,			
                125:0.5,
                0:0,
            }
    
    return partials[hearts]

def calculate_difference(old, new, list_of_items):
    """
    Calculate the difference between items.
    """
    difference = 0
    for item in list_of_items:
        difference += abs(old[item] - new[item])
    return difference

def get_reward(visited_rooms, info=None, old_info=None, level_state="gamestart"):
    """
    Calculate the reward based on the information provided by the environment.
    """
    if info is None:
        return 0, old_info, visited_rooms

    reward = 0

    if old_info is None:
        old_info = info
        old_info['Hearts'] = get_actual_hearts(old_info['Hearts'])
        return 0, old_info, visited_rooms
    
    # If we load a level state and Link is in the overworld, we get a negative reward.
    if level_state in dungeon_save_states and info['Level'] < 1:
        reward = reward_values['level_state_in_overworld']
    
    reward += calculate_difference(old_info, info, single_pickup_items)
    reward += calculate_difference(old_info, info, multi_pickup_items)

    info['Hearts'] = get_actual_hearts(info['Hearts'])

    if info['Hearts'] != old_info['Hearts']:
        if info['Hearts'] < old_info['Hearts']:
            reward -= old_info['Hearts'] - info['Hearts']
        elif info['Hearts'] > old_info['Hearts']:
            reward += info['Hearts'] - old_info['Hearts']

    if info['Room'] not in visited_rooms:
        visited_rooms[info['Room']] = {}
        reward += reward_values['new_room']
    elif (info['Room'] == old_info['Room'] 
        and (info["Enemies Killed Current Room"] > old_info["Enemies Killed Current Room"]
             or (info["Enemies Killed Current Room"] == 0 and old_info["Enemies Killed Current Room"] == 9))):
        """
        This one is particularly messy, so to explain:
        Zelda has a counter for how many enemies have been killed in the current room.
        The counter resets at 10, so if it's 0 and the old counter was 9, that counts as a kill.
        """
        reward += 10


    # Encourage movement around the room
    if (info['Link X'], info['Link Y']) not in visited_rooms[info['Room']]:
        visited_rooms[info['Room']][(info['Link X'], info['Link Y'])] = 1
        reward += reward_values['movement']
    else:
        reward += reward_values['repeat_state']
        
    if info['Enemies Killed'] != old_info['Enemies Killed']:
        reward += (info['Enemies Killed'] - old_info['Enemies Killed']) * 10


    if info['Deaths'] != old_info['Deaths']:
        if info['Deaths'] > old_info['Deaths']:
            reward -= (info['Deaths'] - old_info['Deaths'])
        elif info['Deaths'] < old_info['Deaths']:
            reward += (old_info['Deaths'] - info['Deaths'])

    old_info = info
    return reward, old_info, visited_rooms
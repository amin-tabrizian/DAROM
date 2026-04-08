import numpy as np
import traci
from .utils import get_absolute_lane_idx_from_y
from .merging import parseAction
def safetyCheck(action, state, delayed, mode):
    min_gap = 10.0
    reaction_time = 1.0
    max_acc = 5.0
    min_acc = -5
    veh_length = 5.0
    dt = 0.1  # time step in seconds (from SUMO config)
    parsed_action = parseAction(action)
    delay_steps = 0
    if delayed:
        # Original observation shape: (31, 3) = (num_entities, feature_dim)
        num_entities = 5
        feature_dim = 3
        obs_size = num_entities * feature_dim  # 93 elements
        if mode == 'all':
            observation = np.array(state[:obs_size]).reshape(num_entities, feature_dim)
            # Extract delay value (last element of state)
            delay_steps = state[-1]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    else:
        observation = state
    ego_id = 'ego'
    ego_x = observation[0][0]
    ego_y = observation[0][1]
    ego_speed = observation[0][2]
    ego_lane = get_absolute_lane_idx_from_y(ego_y)
    # check lane change safety
    if parsed_action[1] == 1:
        target_lane = ego_lane + 1
    elif parsed_action[1] == -1:
        target_lane = ego_lane - 1
    else:
        target_lane = ego_lane
    # Initialize info dictionary to store safety check details
    info = {
        'front_gap': None,
        'back_gap': None,
        'front_stopping_dist': None,
        'back_stopping_dist': None,
        'front_approach_speed': None,
        'back_approach_speed': None,
        'front_relative_speed': None,
        'back_relative_speed': None,
        'front_speed': None,
        'back_speed': None,
        'delay_steps': delay_steps,
        'target_lane': None,
        'safe': True
    }
    
    if ego_lane == -1:
        info['target_lane'] = target_lane
        return action, info
    
    # find closest front and back vehicles in the same lane from the observation
    front_vehicle_state  = None
    back_vehicle_state = None
    potential_front_vehicle_state = None
    potential_back_vehicle_state = None
    min_front_dist = float('inf')
    min_back_dist = float('inf')
    for veh_state in observation[1:]:
        y = veh_state[1] + ego_y
        lane = get_absolute_lane_idx_from_y(y)
        if lane == target_lane:
            if veh_state[0] > 0:
                if veh_state[0] < min_front_dist:
                    min_front_dist = veh_state[0]
                    front_vehicle_state = veh_state
            elif veh_state[0] < 0:
                if abs(veh_state[0]) < min_back_dist:
                    min_back_dist = abs(veh_state[0])
                    back_vehicle_state = veh_state
    
    # Predict front and back vehicle states based on delay
    if delayed and delay_steps > 0:
        # Predict front vehicle position after delay
        if front_vehicle_state is not None:
            # veh_state format: [rel_x, rel_y, rel_velocity]
            # rel_velocity = vehicle_velocity - ego_velocity
            # After delay_steps, the relative x position changes by: rel_velocity * delay_steps * dt
            predicted_rel_x = front_vehicle_state[0] + front_vehicle_state[2] * delay_steps * dt
            if predicted_rel_x < 0:
                potential_back_vehicle_state = np.array([
                    predicted_rel_x,
                    front_vehicle_state[1],
                    front_vehicle_state[2]
                ])
                potential_back_dist = abs(predicted_rel_x)
                front_vehicle_state = None
            else:
                front_vehicle_state = np.array([
                    predicted_rel_x,
                    front_vehicle_state[1],
                    front_vehicle_state[2]
                ])
                min_front_dist = predicted_rel_x

        
        # Predict back vehicle position after delay
        if back_vehicle_state is not None:
            # Same prediction for back vehicle
            predicted_rel_x = back_vehicle_state[0] + back_vehicle_state[2] * delay_steps * dt
            back_vehicle_state = np.array([
                predicted_rel_x,
                back_vehicle_state[1],  # y position doesn't change
                back_vehicle_state[2]   # relative velocity remains the same
            ])
            min_back_dist = abs(predicted_rel_x)
            if predicted_rel_x > 0:
                potential_front_vehicle_state = np.array([
                    predicted_rel_x,
                    back_vehicle_state[1],
                    back_vehicle_state[2]
                ])
                back_vehicle_state = None
                potential_front_dist = predicted_rel_x
        if potential_back_vehicle_state is not None:
            if potential_back_dist < min_back_dist:
                back_vehicle_state = potential_back_vehicle_state
                min_back_dist = potential_back_dist
        if potential_front_vehicle_state is not None:
            if potential_front_dist < min_front_dist:
                front_vehicle_state = potential_front_vehicle_state
                min_front_dist = potential_front_dist
    info['target_lane'] = target_lane
    
    if front_vehicle_state is not None:
        # front_vehicle_state[2] is relative velocity: vehVel_front - egoVel
        # Absolute front vehicle speed:
        front_speed = ego_speed + front_vehicle_state[2]
        # By definition: relative_speed = ego_speed - veh_speed
        # Positive when ego is faster and is closing in on the front vehicle.
        relative_speed = ego_speed - front_speed
        approach_speed = max(relative_speed, 0.0)

        gap = min_front_dist 
        if approach_speed > 0:
            stopping_dist = (approach_speed ** 2) / (2 * abs(min_acc))
        else:
            stopping_dist = 0
        
        # Store front vehicle info
        info['front_gap'] = gap
        info['front_stopping_dist'] = stopping_dist
        info['front_approach_speed'] = approach_speed
        info['front_relative_speed'] = relative_speed
        info['front_speed'] = front_speed
        
        if gap < stopping_dist + min_gap and parsed_action[1] != 0:
            print('Unsafe lane change becuase of front vehicle')
            info['safe'] = False
            return [action[0], 0], info
        elif gap < stopping_dist + min_gap and parsed_action[1] == 0:
            print('Unsafe lane lane keeping becuase of front vehicle')
            info['safe'] = False
            return [-5, 0], info

    if back_vehicle_state is not None:
        # back_vehicle_state[2] is relative velocity: vehVel_back - egoVel
        # Absolute back vehicle speed:
        back_speed = ego_speed + back_vehicle_state[2]
        # By definition: relative_speed = ego_speed - veh_speed
        # For a vehicle behind, we care when it is faster than ego,
        # i.e. when (veh_speed - ego_speed) > 0  <=>  relative_speed < 0.
        relative_speed = ego_speed - back_speed
        approach_speed = max(-relative_speed, 0.0)  # positive when back vehicle is catching up

        gap = min_back_dist 
        if approach_speed > 0:
            stopping_dist = (approach_speed ** 2) / (2 * abs(min_acc))
        else:
            stopping_dist = 0
        
        # Store back vehicle info
        info['back_gap'] = gap
        info['back_stopping_dist'] = stopping_dist
        info['back_approach_speed'] = approach_speed
        info['back_relative_speed'] = relative_speed
        info['back_speed'] = back_speed
        
        if gap < stopping_dist + min_gap and parsed_action[1] != 0:
            print('Unsafe lane change becuase of back vehicle')
            info['safe'] = False
            return [action[0], 0], info
        elif gap < stopping_dist + min_gap and parsed_action[1] == 0:
            print('Unsafe lane lane keeping becuase of back vehicle')
            info['safe'] = False
            return [-5, 0], info
    if ego_x < 150 and parsed_action[1] != 0:
        print('Unsafe lane change becuase of ego position')
        info['safe'] = False
        return [action[0], 0], info
    if get_absolute_lane_idx_from_y(ego_y) == 1 and parsed_action[1] != 0:
        print('Unnecessary lane change to left lane')
        info['safe'] = False
        return [action[0], 0], info

    return action, info
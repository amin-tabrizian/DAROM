import numpy as np
import os, sys
import numpy as np
import optparse
# from inference import model
import gymnasium as gym
from gymnasium import spaces
from .utils import *


maxSteps = 200

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

def isEmergencyBraking():
    for vehID in traci.vehicle.getIDList():
        if traci.vehicle.getAcceleration(vehID) <- 4.5:
            return True
    return False

def parseAction(action):
    acc = np.clip(action[0], -5, 5)
    lc = np.clip(action[1], -5, 5)
    if lc < -5 + 10/3:
        lc = -1
    elif lc > -5 + 10/3 and lc < -5 + 20/3:
        lc = 0
    elif lc > -5 + 20/3:
        lc = 1
    return [acc, lc]

def SetTrafficBehavior(vehicle_name):
    if 'f_1' in vehicle_name:
            traci.vehicle.setTau(vehicle_name, np.random.uniform(0.1, 0.7))
            traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(10, 13))
            traci.vehicle.setSpeedMode(vehicle_name, 32)
    elif 'f_2' in vehicle_name:
        traci.vehicle.setTau(vehicle_name, np.random.normal(0.7, 0.1))
        traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(8, 11))
    elif 'ego' in vehicle_name:
        traci.vehicle.setSpeedMode("ego", 96)
        traci.vehicle.setLaneChangeMode('ego', 0b000000000000)



class Merging(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 10}
    def __init__(self, seed, mode='all', radius = 50, render_mode= None, runSafetyCheck=False):
        self.done = False
        self.observation = []
        self.reward = 0
        self.state = []
        self.ego_inserted = False
        self.egoIdx = 0
        self.seed = seed
        self.radius = radius
        self.size = 31
        self.merged = False
        self.lane_id = ''
        self.lane_name = ''
        self.lane_idx = 0
        self.absolute_lane_idx = 0
        self.parsed_action = None
        self.action = None
        self.runSafetyCheck = runSafetyCheck
        self.last_acc = 0
        self._ahead_state = []
        self._behind_state = []
        if not render_mode == 'human':
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

        self.num_entities = self.size          # 31 = ego + 30 neighbors
        self.feature_dim = 3                   # [x, y, vel]

        self.observation_space = spaces.Box(
            low=-1e3, high=1e3,
            shape=(self.num_entities, self.feature_dim),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

        self.render_mode = render_mode

    

    def reset(self, seed=None, options=None):
        # Seed the environment RNG per Gymnasium API
        print('Resetting environment', 'radius', self.radius)
        super().reset(seed=self.seed)
        self.merged = False
        self.given_merge_reward = False
        self.done = False
        self.ego_inserted = False
        self.counter = 0
        self.last_acc = 0
        self._ahead_state = []
        self._behind_state = []
       

        # Close any existing SUMO connection
        try:
            traci.close()
        except Exception:
            pass

        # Start a new SUMO connection
        HOME = '/home/amin/OnRamp'
        SUMO = '/sumo_files/mergingP.sumocfg'

        try:
            sumo_cmd = [
                self.sumoBinary,
                "-c", HOME + SUMO,
                "--tripinfo-output", "tripinfo.xml",
                "--no-step-log",
                "--step-length", "0.1",
                "--collision.check-junctions",
                "--collision.action", "remove",
                "-Q", "--random"
            ]
            traci.start(sumo_cmd)
        except Exception as e:
            raise e

        
        
        while not self.ego_inserted:
            traci.simulationStep()
            if 'ego' in traci.simulation.getDepartedIDList():
                SetTrafficBehavior('ego')
                self.ego_inserted = True
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
                # print('ego inserted')
                
            for vehicle_name in traci.simulation.getDepartedIDList():
                SetTrafficBehavior(vehicle_name)
        
        self.state = self.observation
        return self.state, {}
    
    def getReward(self):

        # print(self.egoIdx)
        # print('action', action, 'safe_action', safe_action)
        # print(self.state[self.egoIdx])
        # Use ego x-position delta only to produce a scalar distance traveled
        distance_traveled = float(self.observation[self.egoIdx, 0] - self.state[self.egoIdx, 0])
        # print(distance_traveled)
        r = -0.30
        base_penalty = -0.30

        jerk = np.abs(self.parsed_action[0] - self.last_acc)
        jerk_penalty = 1.5 * jerk
        r -= jerk_penalty
        # r = 0
        try:
            progress_reward = 1.5 * distance_traveled
            r += progress_reward
        except Exception:
            progress_reward = 0.0
            pass

        is_safe = safe_gap(self._behind_state, self._ahead_state)

        penalty_behind = 0.0
        penalty_ahead = 0.0

        if not is_safe[0]:
            # Use tanh penalty for behind; use distance to behind vehicle
            behind_dist = abs(self._behind_state[0])
            penalty_behind = 10 * np.tanh(1.0 / (behind_dist + 1e-3))  # strong penalty for small gaps, saturates at 10
            r -= penalty_behind

        if not is_safe[1]:
            # Use tanh penalty for ahead; use distance to ahead vehicle
            ahead_dist = abs(self._ahead_state[0]) 
            penalty_ahead = 10 * np.tanh(1.0 / (ahead_dist + 1e-3))
            r -= penalty_ahead

        collision_penalty = 0.0
        arrived_reward = 0.0
        emergency_stopping_penalty = 0.0
        merge_reward = 0.0
        wrong_lane_change_penalty = 0.0
        max_steps_penalty = 0.0

        if 'ego' in traci.simulation.getCollidingVehiclesIDList():
            print("Colliding")
            collision_penalty = 1000
            r -= collision_penalty
        elif 'ego' in traci.simulation.getArrivedIDList():
            print("Arrived")
            arrived_reward = 500
            r += arrived_reward
        elif 'ego' in traci.simulation.getEmergencyStoppingVehiclesIDList():
            print("Emergency Stopping")
            emergency_stopping_penalty = 1000
            r -= emergency_stopping_penalty
        elif self.merged and not self.given_merge_reward:
            print("Merge reward given.")
            self.given_merge_reward = True
            merge_reward = 500
            r += merge_reward
        elif self.parsed_action[1] != 0:
            # print("Wrong lane change")
            wrong_lane_change_penalty = 5
            r -= wrong_lane_change_penalty
        elif traci.simulation.getTime() > maxSteps:
            print("Max Steps")
            max_steps_penalty = 1000
            r -= max_steps_penalty

        # print(f"Reward components:"
        #       f"\n  base_penalty: {base_penalty}"
        #       f"\n  progress_reward: {progress_reward}"
        #       f"\n  jerk_penalty: {-jerk_penalty}"
        #       f"\n  penalty_behind: {-penalty_behind}"
        #       f"\n  penalty_ahead: {-penalty_ahead}"
        #       f"\n  collision_penalty: {-collision_penalty}"
        #       f"\n  arrived_reward: {arrived_reward}"
        #       f"\n  emergency_stopping_penalty: {-emergency_stopping_penalty}"
        #       f"\n  merge_reward: {merge_reward}"
        #       f"\n  wrong_lane_change_penalty: {-wrong_lane_change_penalty}"
        #       f"\n  max_steps_penalty: {-max_steps_penalty}"
        #       f"\n  TOTAL (BEFORE SCALING): {r}")

        # print("reward", r, "action", action, "lane", self.lane_id, "merged", self.merged)
        return 0.1*r    
    
    def _get_obs(self):

        # Get ego vehicles' pos and vel
        egoPos = traci.vehicle.getPosition('ego')
        egoVel = traci.vehicle.getSpeed('ego')

        if egoPos[0] < 0:
            raise Exception()
        ego = [list(egoPos) + [egoVel]]

        # Get surrounding vehicles pos and vel
        intruders = []
        self._ahead_state = [] # closest ahead vehicle state
        self._behind_state = [] # closest behind vehicle state
        closest_ahead = None
        closest_behind = None
        min_ahead_dist = float('inf')
        min_behind_dist = float('inf')
        for vehID in traci.vehicle.getIDList():
            if vehID == 'ego':
                continue
            vehPos = traci.vehicle.getPosition(vehID)
            distance = getDistance(vehPos, egoPos) # distance between vehicle and ego everything is absolute coordinates.
            if distance <= self.radius:
                vehVel = traci.vehicle.getSpeed(vehID)
                relState = [vehPos[0] - egoPos[0], vehPos[1] - egoPos[1], vehVel - egoVel]
                # print('Distance is less than radius', distance, 'for vehicle', vehID, 'relState', relState)
                intruders.append(relState)
                # Determine if ahead or behind by x position relative to ego
                if relState[0] > 0:  # ahead
                    if relState[0] < min_ahead_dist:
                        min_ahead_dist = relState[0]
                        closest_ahead = relState
                elif relState[0] < 0:  # behind
                    if abs(relState[0]) < min_behind_dist:
                        min_behind_dist = abs(relState[0])
                        closest_behind = relState
        self._ahead_state = closest_ahead if closest_ahead is not None else [0.0, 0.0, 0.0]
        self._behind_state = closest_behind if closest_behind is not None else [0.0, 0.0, 0.0]
        # Sort intruders first by y position, then by x position, both ascending
        intruders = intruders[:self.num_entities - 1] # ego will take the last slot in the observation space.
        intruders = sorted(intruders, key=lambda x: (x[1], x[0])) # sort intruders by y position, then by x position, both ascending
        
    
        while len(intruders) < self.num_entities - 1:
            intruders.append([0.0, 0.0, 0.0])
        # print('Intruders', intruders, 'ego', ego)
        obs = np.array(ego + intruders, dtype=np.float32)
        return obs
        

    def step(self, action):
        terminated = False
        truncated = False
        info = {'message': ''}
        self.lane_id = traci.vehicle.getLaneID('ego')
        self.lane_name = extract_lane_name(self.lane_id)
        try:
            self.lane_idx = traci.vehicle.getLaneIndex('ego')
        except Exception:
            self.lane_idx = 0
        self.absolute_lane_idx = get_absolute_lane_idx_from_vehID('ego')


        parsed_action = parseAction(action)
        self.action = action
        if self.runSafetyCheck:
            try:
                safe_action = self.safetyCheck(parsed_action)
                parsed_action = safe_action.copy()
                self.action = parsed_action
            except Exception as e:
                print('Error in safetyCheck', e)

        if not self.done:
            if 'E3' == self.lane_name and self.merged:
                min_lane_number = 1
            else:
                min_lane_number = 0

            max_lane_number = 5 if 'E3' == self.lane_name else 4
            if 'E3'  == self.lane_name or 'E0' == self.lane_name:
                if parsed_action[1] == 1:
                    self.target_lane_idx= min(self.lane_idx + 1, max_lane_number)
                    traci.vehicle.changeLane('ego', self.target_lane_idx, 0)
                elif parsed_action[1] == -1:
                    self.target_lane_idx= max(self.lane_idx - 1, min_lane_number)
                    traci.vehicle.changeLane('ego', self.target_lane_idx, 0)

            for vehicle_name in traci.simulation.getDepartedIDList():
                SetTrafficBehavior(vehicle_name)
            traci.vehicle.setAcceleration('ego', parsed_action[0], 1)
            if traci.vehicle.getSpeed('ego') > 32:
                traci.vehicle.setSpeed('ego', 32)
            # traci.vehicle.setSpeed('ego', min(32, traci.vehicle.getSpeed('ego')))
            self.counter += 1
            traci.simulationStep()

            try:
                self.lane_id = traci.vehicle.getLaneID('ego')
                self.lane_name = extract_lane_name(self.lane_id)
                self.absolute_lane_idx = get_absolute_lane_idx_from_vehID('ego')
            except:
                print('Error in getting lane id')
            self.parsed_action = parsed_action
            if 'E3' == self.lane_name and self.absolute_lane_idx != 0 and not self.merged:
                print('Merged')
                self.merged = True

            if 'ego' in traci.simulation.getCollidingVehiclesIDList():
                info['message'] = 'Collision'
                terminated = True
                self.done = True
                self.observation = self.state
            elif 'ego' in traci.simulation.getArrivedIDList():
                terminated = True
                self.done = True
                self.observation = self.state
                info['message'] = 'Arrived'
            elif  traci.simulation.getTime() > maxSteps:
                print('Terminating episode since it exceeded maximum time.')
                info['message'] = 'Max Steps'
                truncated = True
                self.done = True
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
                
            elif 'ego' in traci.simulation.getEmergencyStoppingVehiclesIDList():
                self.done = True
                terminated = True
                info['message'] = 'Emergency Stopping'
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
                
            else:
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
               
        self.reward = self.getReward()
        self.state = self.observation.copy()
        self.last_acc = self.parsed_action[0]
        if self.done:
            try:
                traci.close()
            except Exception:
                pass
        
        info.update({'lane': self.lane_id, 'action': action})
        return self.observation, self.reward, terminated, truncated, info


    def render(self):
        # Gymnasium render contract: return frame for rgb_array, None for human
        if self.render_mode == "human":
            # SUMO-GUI handles visualization when using sumo-gui; nothing to draw here
            return None
        return None

    def close(self):
        try:
            traci.close()
        except Exception:
            print('Error in closing SUMO')

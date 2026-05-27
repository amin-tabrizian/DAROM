import numpy as np
import torch
import traci
import random
from collections.abc import Iterable


minBatchSize = 1000
def getDistance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def flatten(xss):
    flat_list = []
    for xs in xss:
        if isinstance(xs, Iterable):
            for x in xs:
                if isinstance(x, Iterable):
                    for y in x:
                        flat_list.append(round(y, 2))
                else:
                    flat_list.append(round(x, 2))
        else:
            flat_list.append(round(xs, 2))
    return flat_list

def time_headway(ego_state, intruder_state, ego_front):
    # ego_state: [x, y, v], intruder_state: [x_rel, y_rel, v_rel] is relative to ego
    # ego_front: True if ego is in front of intruder, False if ego is behind intruder
    if ego_front:
        return (-intruder_state[0]) / (ego_state[2] + 0.001)
    else:
        return (intruder_state[0]) / (intruder_state[2] + ego_state[2] + 0.001)





class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
     
def eval_policy(policy, eval_env, seed, eval_episodes=10):

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward[0]

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def get_absolute_lane_idx_from_vehID(vehID):
    y_to_lane_idx = {
        -1.6: 5,
        -4.8: 4,
        -8.0: 3,
        -11.2: 2,
        -14.4: 1,
        -17.6: 0
    }
    y = traci.vehicle.getPosition(vehID)[1]
    # Find the closest key within a small tolerance to avoid floating point errors
    for key in y_to_lane_idx:
        if abs(y - key) < 0.01:
            return y_to_lane_idx[key]
    return -1
def get_absolute_lane_idx_from_y(y):
    y_to_lane_idx = {
        -1.6: 5,
        -4.8: 4,
        -8.0: 3,
        -11.2: 2,
        -14.4: 1,
        -17.6: 0
    }
    # Find the closest key within a small tolerance to avoid floating point errors
    for key in y_to_lane_idx:
        if abs(y - key) < 0.5:
            return y_to_lane_idx[key]
    return -1

def extract_lane_name(lane_str):
    """
    Extracts the lane or junction name from a SUMO lane string.
    Examples:
        ":J2_0_0" -> "J2"
        "E4_0"    -> "E4"
    """
    if lane_str.startswith(":"):
        # Junction format: ":J2_0_0" -> "J2"
        parts = lane_str.split("_")
        if parts:
            return parts[0][1:]  # remove leading ":"
    else:
        # Lane format: "E4_0" -> "E4"
        parts = lane_str.split("_")
        if parts:
            return parts[0]
    return lane_str  # fallback: return as is


def safe_gap(behindState, aheadState):
    behind_sum = np.sum(behindState)
    ahead_sum = np.sum(aheadState)
    safe_gap = 20
    is_safe = [False, False]

    # If behind vehicle is "sum 0" then it's safe
    if behind_sum == 0:
        is_safe[0] = True
    elif np.abs(behindState[1]) > 0.1:
        is_safe[0] = True
    elif np.abs(behindState[0]) > safe_gap:
        is_safe[0] = True

    # If ahead vehicle is "sum 0" then it's safe
    if ahead_sum == 0:
        is_safe[1] = True
    elif np.abs(aheadState[1]) > 0.1:
        is_safe[1] = True
    elif np.abs(aheadState[0]) > safe_gap:
        is_safe[1] = True
    
    
    return is_safe

class PredictorDataset:
    def __init__(self):
        self.waiting = []  # samples that are waiting for their target
        self.ready = []    # samples with known targets, used for training
        self.masked_augmented_state = None
        self.ready_samples = []

    def update(self, augmented_state, true_state, current_step, current_observation_delay, state_shape, action_shape):
        augmented_state = flatten(augmented_state)
        # print('Augmented state:', augmented_state)
        action_list = augmented_state[state_shape: -1]
        received_observation_delay = augmented_state[-1]
        if received_observation_delay == 0:
            action_list_masked = np.zeros_like(action_list)
        else:
            # print('A/ction list:', action_list)
            action_list_masked = np.zeros_like(action_list)
            action_list_masked[0:received_observation_delay*action_shape] = action_list[0:received_observation_delay*action_shape]
        self.masked_augmented_state = np.concatenate((augmented_state[0:state_shape], action_list_masked, [received_observation_delay]))
        self.waiting.append({
            "input": torch.tensor(self.masked_augmented_state, dtype=torch.float32),
            "target": torch.tensor(true_state, dtype=torch.float32),
            "ready_step": current_step + current_observation_delay
        })
        self.ready_samples = []
        new_waiting = []
        for sample in self.waiting:
            if sample["ready_step"] == current_step:
                self.ready_samples.append({"input": sample["input"], "target": sample["target"]})
            else:
                new_waiting.append(sample)
        self.ready.extend(self.ready_samples)
        self.waiting = new_waiting
        # print('Ready samples:', ready_samples, '\n Waiting samples:', self.waiting)
        

    def get_batch(self, batch_size):
        if len(self.ready) < minBatchSize:
            return None
        batch = random.sample(self.ready, batch_size)
        inputs = torch.stack([s["input"] for s in batch])
        targets = torch.stack([s["target"] for s in batch])
        return inputs, targets


def fine_tune_predictor(model, model_path, predictor_dataset: 'PredictorDataset', epochs, batch_size, lr, device):
        """
        Fine-tune the given model using the provided PredictorDataset.
        Saves the model's weights to model_path after fine-tuning.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        if len(predictor_dataset.ready) < batch_size:
            raise ValueError("Not enough samples in predictor_dataset.ready to fine-tune.")

        for epoch in range(epochs):
            model.train()
            random.shuffle(predictor_dataset.ready)
            total_loss = 0.0
            num_batches = 0
            for i in range(0, len(predictor_dataset.ready), batch_size):
                batch_samples = predictor_dataset.ready[i:i+batch_size]
                if len(batch_samples) < batch_size:
                    continue
                inputs = torch.stack([s["input"] for s in batch_samples]).to(device)
                targets = torch.stack([s["target"] for s in batch_samples]).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / max(1, num_batches)
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Fine-tune Loss: {avg_loss:.6f}")

        # Save only the model's weights to the given path
        # Save the entire checkpoint for a transformer model, raising errors if any attribute is missing
        required_attrs = ['input_dim', 'output_dim', 'dropout',
            'model_state_dict', 'model_dim', 'num_heads', 'num_layers'
        ]

        checkpoint = {}

        for attr in required_attrs:
            if attr == 'model_state_dict':
                checkpoint[attr] = model.state_dict()
            else:
                if not hasattr(model, attr):
                    raise AttributeError(f"Transformer model is missing required attribute '{attr}'")
                checkpoint[attr] = getattr(model, attr)

        # Scalers from predictor_dataset
        if not hasattr(predictor_dataset, 'feature_scaler') or not hasattr(predictor_dataset, 'target_scaler'):
            raise AttributeError("predictor_dataset must have 'feature_scaler' and 'target_scaler' attributes")
        checkpoint['feature_scaler'] = predictor_dataset.feature_scaler
        checkpoint['target_scaler'] = predictor_dataset.target_scaler

        torch.save(checkpoint, model_path)

        return model
# INSERT_YOUR_CODE
def get_user_action():
    import sys
    # print("Press arrow key (→ [5,0], ← [-5,0], ↑ [0,5], ↓ [0,-5], Enter [0,0]): ", end="", flush=True)
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == '\x1b':  # Arrow keys start with ESC
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'C':   # Right arrow
                    action = [5, 0]
                elif ch3 == 'D': # Left arrow
                    action = [-5, 0]
                elif ch3 == 'A': # Up arrow
                    action = [0, 5]
                elif ch3 == 'B': # Down arrow
                    action = [0, -5]
                else:
                    action = [0, 0]
            else:
                action = [0, 0]
        elif ch1 == '\r' or ch1 == '\n':  # Enter
            action = [0, 0]
        else:
            action = [0, 0]
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # print(f" Chosen action: {action}")
        return action
    except Exception:
        # On Windows or fallback, use input()
        inp = input()
        # Accept wasd/arrows/enter as text
        if inp.lower() in ["right", "d"]:
            return [5, 0]
        elif inp.lower() in ["left", "a"]:
            return [-5, 0]
        elif inp.lower() in ["up", "w"]:
            return [0, 5]
        elif inp.lower() in ["down", "s"]:
            return [0, -5]
        elif inp == "":
            return [0, 0]
        else:
            return [0, 0]
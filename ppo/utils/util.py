import numpy as np
import math
import time
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Literal, Optional, Union
from collections import defaultdict
from contextlib import contextmanager
from termcolor import cprint
import cv2


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


class TimingManager:
    def __init__(self,):
        self.timing_stats = defaultdict(float)
        self.call_counts = defaultdict(int)
        self.start_time = time.time()
        
    @contextmanager
    def timer(self, name, debug=False):
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timing_stats[name] += elapsed
            self.call_counts[name] += 1
            if debug:
                print(f"Timing for {name}: {elapsed:.2f} seconds")

    def get_log(self):
        avg_times = {
            f"timing/{k}": self.timing_stats[k]# / self.call_counts[k] 
            for k in self.timing_stats
        }
        total_time = sum(self.timing_stats.values())
        time_percentages = {    # here we do not consider the call counts
            f"timing/percent_{k}": (v / total_time) * 100 
            for k, v in self.timing_stats.items()
        }
        # stats = {**avg_times, **time_percentages}
        stats = {**time_percentages}
        return stats

    def close(self):
        total_time = time.time() - self.start_time
        cprint(f"Total time: {total_time:.2f} seconds", "green")
        return total_time

def add_info_board(img, **kwargs):
    """
    Add a white information board to the right of an image with flexible content.
    
    Args:
        img: Input image (numpy array)
        **kwargs: Variable key-value pairs to display on the board
        
    Returns:
        combined_img: Image with info board added
    """
    # Fixed parameters
    board_width = 280
    board_height = img.shape[0]
    line_height = 15
    character_limit = 35
    margin_left = img.shape[1] + 10
    
    # Create white board and combine with image
    board = np.ones((board_height, board_width, 3), dtype=np.uint8) * 255
    combined_img = np.hstack((img, board))
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_color = (0, 0, 0)
    
    # Draw title
    # cv2.putText(combined_img, "States", (margin_left, 30),
    #             font, font_scale + 0.0, text_color, 2, cv2.LINE_AA)
    # y_position = 50     # the y position of the first line
    y_position = 10
    
    # Process each key-value pair
    for key, value in kwargs.items():
        if value is None:
            continue
        # Draw key
        cv2.putText(combined_img, f"{key}:", (margin_left, y_position),
                    font, font_scale, text_color, 1, cv2.LINE_AA)
        y_position += line_height
        
        # Convert value to string and split into lines
        value_str = str(value)
        lines = []
        words = value_str.split()
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > character_limit:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw value lines
        for line in lines:
            cv2.putText(combined_img, line, (margin_left, y_position),
                       font, font_scale, text_color, 1, cv2.LINE_AA)
            y_position += line_height
            
        # Add blank line between different inputs
        # y_position += line_height
        
        # Check if we're running out of vertical space
        if y_position >= board_height - line_height:
            break
            
    return combined_img
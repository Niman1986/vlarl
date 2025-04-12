from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, Tuple, Union, List, TypedDict
import numpy as np

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')


class EnvOutput(TypedDict):
    pixel_values: List[np.ndarray]
    prompts: List[str]

class BaseEnv(ABC, Generic[ObsType, ActionType]):
    """
    Abstract base class for Reinforcement Learning environments.
    
    This class defines the standard interface that should be implemented by
    all RL environments. It supports both standard RL tasks and multi-modal
    tasks (e.g., vision-language tasks).
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the environment.
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def reset(self) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Tuple[ObsType, Dict[str, Any]]: Initial observation and info dictionary
        """
        pass
    
    @abstractmethod
    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action (ActionType): Action to be executed
            
        Returns:
            Tuple containing:
                - observation (ObsType): Environment observation
                - reward (float): Reward from the action
                - done (bool): Whether episode ended naturally
                - info (Dict[str, Any]): Additional information
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Any:
        """
        Define the action space of the environment.
        
        Returns:
            Any: Description of the action space
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """
        Define the observation space of the environment.
        
        Returns:
            Any: Description of the observation space
        """
        pass
    
    def seed(self, seed: int = None) -> None:
        """
        Set the seed for the environment's random number generator.
        
        Args:
            seed (int, optional): The seed value
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def unwrapped(self):
        """
        Returns the base environment object.
        
        Useful for cases of wrapped environments
        """
        return self

"""
ENVIRONMENR FACTORY
-------------------

This module includes attributes and methods to randomly generate network environments.

Requirements: 
- The grid cannot be larger than 640,000 m^2
- The grid cannot be smaller than 40,000 m^2

@author: Giuliana Emberson
@date: 7th of May 2026

"""

import random
from typing import Optional, Tuple
from dataclasses import dataclass


# Grid size bounds (meters)
def random_grid_size(bounds_min: float, bounds_max: float) -> Tuple[float, float]:
    grid_width = random.uniform(bounds_min, bounds_max)
    grid_height = random.uniform(bounds_min, bounds_max)
    return (grid_width*2, grid_height*2)

@dataclass
class Environment:
    x_domain: Tuple[float, float]  # meters
    y_range: Tuple[float, float]   # meters
    width: float                   # meters
    height: float                  # meters
    area: float                    # m^2
    env_type: str                  # "indoor" or "outdoor"
    
    def __init__(self, *, env_type: Optional[str] = None):
        # Use larger environment bounds so that realistic 5 GHz coverage radius (80–230 m) requires multiple antennas to cover the full map.
        #   - Min: 100 (200×200 = 40,000 m^2)  → 100 for the +(x,y) and 100 for -(x,y)
        #   - Max: 400 (800×800 = 640,000 m^2) → 400 for the +(x,y) and 400 for -(x,y)
        #
        # This keeps the environments large enough to need more than one antenna while keeping the scenario computationally manageable.
        bounds_min = 100   # meters
        bounds_max = 400   # meters

        self.width, self.height = random_grid_size(bounds_min, bounds_max)

        self.x_domain = (-self.width / 2, self.width / 2)
        self.y_range = (-self.height / 2, self.height / 2)
        
        self.area = self.width * self.height
        self.env_type = env_type or random.choice(["indoor", "outdoor"]) # this will affect the type of antenna coverage range that we select in `antenna_factory.py`

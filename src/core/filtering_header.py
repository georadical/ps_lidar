"""
Filtering Module

Provides functions for point cloud filtering including noise removal and understory separation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.spatial import cKDTree
import warnings

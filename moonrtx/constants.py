"""
Constants used across the Moon renderer modules.
"""

# Scene geometry
MOON_FILL_FRACTION = 0.9    # Moon fills 90% of window height (5% margins top/bottom)
SUN_RADIUS = 11             # affects Moon surface illumination
MOON_RADIUS = 10.0          # Radius of Moon sphere in scene units

# Colors
GRID_COLOR = [0.50, 0.50, 0.50]
PIN_COLOR = [1.0, 0.0, 0.0]

# Line radii
GRID_LINE_RADIUS = 0.006    # Thin lines for grid
GRID_LABEL_RADIUS = 0.012   # Slightly thicker lines for grid labels
STANDARD_LABEL_RADIUS = 0.008  # Standard feature label thickness
SPOT_LABEL_RADIUS = 0.008   # Spot feature label thickness
PIN_LABEL_RADIUS = 0.012    # Pin digit label thickness

# Camera
CAMERA_TYPE = "Pinhole"

# View orientation modes for different telescope configurations
# Each mode specifies: (vertical_flip, horizontal_flip)
# vertical_flip=True means S is up (N is down)
# horizontal_flip=True means E is left (W is right)
ORIENTATION_NSWE = "NSWE"  # Default: N up, S down, W left, E right
ORIENTATION_NSEW = "NSEW"  # N up, S down, E left, W right (horizontal flip)
ORIENTATION_SNEW = "SNEW"  # S up, N down, E left, W right (both flips so same as 180° rotation)
ORIENTATION_SNWE = "SNWE"  # S up, N down, W left, E right (vertical flip)

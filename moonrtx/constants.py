"""
Constants used across the Moon renderer modules.
"""

# Scene geometry
MOON_FILL_FRACTION = 0.9    # Moon fills 90% of window height (5% margins top/bottom)
# Real Sun angular radius ≈ 0.267°.  SUN_RADIUS/SUN_LIGHT_DISTANCE = tan(0.267°) ≈ 0.00466.
# SUN_LIGHT_DISTANCE = 10 / tan(0.267°) ≈ 2146 gives physically accurate angular size and
# parallel rays (divergence across Moon disk ≈ 0.27°, vs. ~0.001° physical — acceptable).
# PlotOptiX color is radiance (per sphere area), so irradiance ∝ (radius/distance)².
# SUN_BRIGHTNESS_SCALE compensates for the increased distance vs. the original 100-unit baseline.
SUN_RADIUS = 10
# SUN_LIGHT_DISTANCE = 2146       # (physically accurate)
SUN_LIGHT_DISTANCE = 1300         # (physically not accurate but with less meshes visible comparing to 2146)
SUN_BRIGHTNESS_SCALE = (SUN_LIGHT_DISTANCE / 100.0) ** 2
MOON_RADIUS = 10.0          # Radius of Moon sphere in scene units

# Colors
GRID_COLOR = [0.50, 0.50, 0.50]
PIN_COLOR = [1.0, 0.0, 0.0]

CAMERA_NAME = "cam1"
LIGHT_NAME = "sun"
MOON_OBJECT_NAME = "moon"

# Line radii
GRID_LINE_RADIUS = 0.006    # Thin lines for grid
GRID_LABEL_RADIUS = 0.012   # Slightly thicker lines for grid labels
STANDARD_LABEL_RADIUS = 0.008  # Standard feature label thickness
SPOT_LABEL_RADIUS = 0.008   # Spot feature label thickness
PIN_LABEL_RADIUS = 0.012    # Pin digit label thickness

# View orientation modes for different telescope configurations
# Each mode specifies: (vertical_flip, horizontal_flip)
# vertical_flip=True means S is up (N is down)
# horizontal_flip=True means E is left (W is right)
ORIENTATION_NSWE = "NSWE"  # Default: N up, S down, W left, E right
ORIENTATION_NSEW = "NSEW"  # N up, S down, E left, W right (horizontal flip)
ORIENTATION_SNEW = "SNEW"  # S up, N down, E left, W right (both flips so same as 180° rotation)
ORIENTATION_SNWE = "SNWE"  # S up, N down, W left, E right (vertical flip)

# View orientation modes for different telescope configurations
# Each mode specifies: (vertical_flip, horizontal_flip)
# vertical_flip=True means S is up (N is down)
# horizontal_flip=True means E is left (W is right)
VIEW_ORIENTATION_NSWE = "NSWE"  # Default: N up, S down, W left, E right
VIEW_ORIENTATION_NSEW = "NSEW"  # N up, S down, E left, W right (horizontal flip)
VIEW_ORIENTATION_SNEW = "SNEW"  # S up, N down, E left, W right (both flips so same as 180° rotation)
VIEW_ORIENTATION_SNWE = "SNWE"  # S up, N down, W left, E right (vertical flip)

VIEW_ORIENTATIONS = [VIEW_ORIENTATION_NSWE, VIEW_ORIENTATION_NSEW, VIEW_ORIENTATION_SNEW, VIEW_ORIENTATION_SNWE]

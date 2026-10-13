"""Dataset-wide constants."""

# Capacitor type prefixes used in the raw folder names (e.g. "C05 VS-W").
CAPACITOR_TYPES = ("C05", "C10", "W18", "Y18", "Z10", "Z18", "Z33")

# Full-resolution camera frame.
IMAGE_WIDTH = 5472
IMAGE_HEIGHT = 3648

# Annotation labels.
MASTER_BOX = "master_box"  # outline of the whole capacitor tray (region detector)
NUMB_BOX = "numb_box"  # capacitor with printed number
BLACK_BOX = "black_box"  # black capacitor
NOT_AVAILABLE = "Not_Available"  # cell without a known defect label

# Image ids whose capacitors are black. Union of the lists used in the old notebooks.
BLACK_BOX_IMAGE_IDS = frozenset(
    {"019", "053", "066", "067", "068", "134", "157", "198", "259", "260", "261", "309", "310"}
)

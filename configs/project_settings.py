# File: configs/project_settings.py

# The order here defines the integer ID. Background is typically 0.

# SUIM Dataset 8 Classes
# Class ID: (Class Name, RGB_Color_for_Visualization)
# RGB color codes are binary (0=0, 1=255) for R, G, B components.

CLASS_MAPPING = {
    "Background waterbody": (0, [0, 0, 0]),           # Black (000)
    "Human divers": (1, [0, 0, 255]),                  # Blue (001)
    "Plants/sea-grass": (2, [0, 255, 0]),             # Green (010)
    "Wrecks/ruins": (3, [0, 255, 255]),                # Cyan (011)
    "Robots/instruments": (4, [255, 0, 0]),            # Red (100)
    "Reefs and invertebrates": (5, [255, 0, 255]),     # Magenta (101)
    "Fish and vertebrates": (6, [255, 255, 0]),        # Yellow (110)
    "Sand/sea-floor (& rocks)": (7, [255, 255, 255])   # White (111)
}

# Reverse mapping for quick lookup from ID to name
ID_TO_CLASS = {v[0]: k for k, v in CLASS_MAPPING.items()}

# Mapping from ID to visualization RGB color
ID_TO_COLOR = {v[0]: v[1] for k, v in CLASS_MAPPING.items()}

# Define a default color for unknown/unmapped labels, usually black (background)
UNKNOWN_COLOR = [0, 0, 0]

# Mapping function for JSON parsing (if applicable, not directly used by SUIM masks)
def get_class_id(label_string):
    # This function expects the full class name string as input
    # It returns the integer ID for that class.
    # If the label_string is not found, it defaults to background (ID 0).
    return CLASS_MAPPING.get(label_string, (0, UNKNOWN_COLOR))[0]
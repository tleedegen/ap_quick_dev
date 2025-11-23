""" Intended as a prototype script to update wall geometry jsons in batch. Wan not really used."""
import xlwings as xw
import json

# Configuration: Adjust these values as needed
import os

# Convert SharePoint/OneDrive path to local path
FILE_PATH = r"C:\Users\djmiller\OneDrive - Degenkolb Engineers\AnchorPro\scripts\JSON Updater.xlsx"
if "~" in FILE_PATH:
    FILE_PATH = os.path.expanduser(FILE_PATH)  # Resolves OneDrive symbolic paths  # Path to your Excel file

INPUT_SHEET_NAME = "INPUT"  # Sheet containing JSON data
OUTPUT_SHEET_NAME = "OUTPUT"  # Sheet where updated JSONs will be written
COLUMN_NAME = "Old JSON"  # Column containing JSON strings

# Update Early version to B1.0
base_updates_b1_0 = [
    ("straps", {'X':[],'Y':[],'Z':[], 'DX':[], 'DY':[], 'DZ':[], 'strap': 'null'}, ["element"]),
    ("release_zn", [], ["layout"])
]

# Update B1.0 to B1.1
base_updates_b1_1 = [("xc",0,[])]

# Update B2.1.X to B2.2.X (Wall Geometry)
wall_updates_b2_2 = [("omit_bracket_output", False,[]),
           ]

# Update v4.1.0 to v5.X.X (Wall Geometry)
wall_updates_v4_2 = []
# REMOVALS = [("attachment_offset",[])]

# UPDATE v4.1.0 to v5.X.X (Base Geometry)
REMOVALS = []
base_updates_v4_2 = [("t_plate", 0, ["element"]),
           ("fy", 0, ["element"])]

# DVC Update
UPDATES = base_updates_b1_0 + base_updates_b1_1 + base_updates_v4_2

def update_element(json_dict):
    # Remove Any Items
    for key_to_remove, nested_path in REMOVALS:
        target = json_dict  # Start at the dictionary level

        # Traverse the nested path to find the correct dictionary
        for key in nested_path:
            if isinstance(target, dict) and key in target:
                target = target[key]
            else:
                break  # Stop if path is invalid
        if key_to_remove in target:
            del target[key_to_remove]

    # Apply all updates
    for new_key, new_val, nested_path in UPDATES:
        target = json_dict  # Start at the dictionary level

        # Traverse the nested path to find the correct dictionary
        for key in nested_path:
            if isinstance(target, dict) and key in target:
                target = target[key]
            else:
                break  # Stop if path is invalid

        # Ensure the target is a dictionary before adding the new key
        if isinstance(target, dict):
            target[new_key] = new_val



def update_json(json_str):
    """Updates a JSON string by adding new key-value pairs in specified nested dictionaries."""
    try:
        parsed_json = json.loads(json_str)  # Parse JSON

        # Ensure the root is a list
        if isinstance(parsed_json, list):
            for item in parsed_json:
                update_element(item)
        elif isinstance(parsed_json, dict):
            update_element(parsed_json)

        return json.dumps(parsed_json, ensure_ascii=False)  # Convert back to string
    except json.JSONDecodeError:
        return json_str  # Return as-is if not valid JSON


# Open the workbook
wb = xw.Book(FILE_PATH)

# Reference the input sheet
ws_input = wb.sheets[INPUT_SHEET_NAME]
ws_output = wb.sheets[OUTPUT_SHEET_NAME]

# Find the header row and the JSON column index
headers = ws_input.range("A1").expand("right").value
json_col_index = headers.index(COLUMN_NAME) + 1  # 1-based index

# Read JSON data from the input sheet (excluding the header)
json_data = ws_input.range((2, json_col_index), ws_input.range((2, json_col_index)).end("down")).value

# Process JSON updates
updated_json_data = [update_json(str(json_str)) for json_str in json_data]

# Write updated JSON data to the output sheet (same column position)
ws_output.range("A1").value = headers  # Copy headers
ws_output.range((2, json_col_index)).value = [[json] for json in updated_json_data]

print(f"Updated JSONs written to '{OUTPUT_SHEET_NAME}' in {FILE_PATH}. Workbook remains open.")


import os


# Function to read file paths from a list file
def get_file_paths_from_list(list_filepath):
    if not os.path.exists(list_filepath):
        raise FileNotFoundError(
            f"File list not found: {list_filepath}. Please ensure the merged preprocessing script ran."
        )
    with open(list_filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]

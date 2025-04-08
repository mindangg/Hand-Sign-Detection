import os
import string

# Parent folder
parent_folder = "data"
os.makedirs(parent_folder, exist_ok=True)

# Create subfolders 0-9
for i in range(10):
    folder_path = os.path.join(parent_folder, str(i))
    os.makedirs(folder_path, exist_ok=True)

# Create subfolders A-Z
for letter in string.ascii_uppercase:
    folder_path = os.path.join(parent_folder, letter)
    os.makedirs(folder_path, exist_ok=True)

print("Folders 0-9 and A-Z created inside 'data/' successfully.")

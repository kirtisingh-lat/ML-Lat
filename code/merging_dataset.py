import os
import shutil

# Provide the full paths to your directories here
base_dir = '/home/ss/Kirti/lat/new_merged/val'  # Change this to your base directory

# List of source folders for images and labels
image_folders = ['images', 'images2', 'images3', 'images4']
label_folders = ['labels', 'labelsobb2', 'labelsobb3', 'labelsobb4']

# Destination folders
image_dest = os.path.join(base_dir, 'img')
label_dest = os.path.join(base_dir, 'labels_combined')

# Create destination directories if they don't exist
os.makedirs(image_dest, exist_ok=True)
os.makedirs(label_dest, exist_ok=True)

# Function to move contents of folders to destination
def move_files(src_folders, dest_folder):
    for folder in src_folders:
        src_folder_path = os.path.join(base_dir, folder)
        if os.path.exists(src_folder_path):
            for item in os.listdir(src_folder_path):
                src_path = os.path.join(src_folder_path, item)
                dest_path = os.path.join(dest_folder, item)
                
                # Check if it's a file and move it
                if os.path.isfile(src_path):
                    shutil.move(src_path, dest_path)
                elif os.path.isdir(src_path):
                    # If it's a directory, move all its contents
                    for sub_item in os.listdir(src_path):
                        shutil.move(os.path.join(src_path, sub_item), dest_folder)

# Move image files
move_files(image_folders, image_dest)

# Move label files
move_files(label_folders, label_dest)

print("Files have been moved successfully.")

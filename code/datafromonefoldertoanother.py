import os
import shutil

def transfer_files(source_dir, dest_dir):
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    # Create destination directory if it does not exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Iterate through all files in the source directory
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        # Check if it's a file and transfer it
        if os.path.isfile(source_path):
            shutil.move(source_path, dest_path)
            print(f"Moved: {source_path} to {dest_path}")
        elif os.path.isdir(source_path):
            # If it's a directory, move the directory recursively
            transfer_files(source_path, dest_path)
            print(f"Moved directory: {source_path} to {dest_path}")

# Example usage
source_folder = '/home/ss/Kirti/lat/datasets/people_only/labels3/train'  # Replace with the source folder path
destination_folder = '/home/ss/Kirti/lat/datasets/people_only/labels/train'  # Replace with the destination folder path

# Call the function
transfer_files(source_folder, destination_folder)
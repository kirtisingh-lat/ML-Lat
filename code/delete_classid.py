import os

def delete_labels_with_class_id(folder_path, class_id_to_delete):
    # Define the paths for the subfolders
    subfolders = ['train', 'val', 'test']
    
    # Initialize a counter for total deleted labels
    total_deleted_labels = 0

    # Traverse through each subfolder
    for subfolder in subfolders:
        folder = os.path.join(folder_path, subfolder)
        
        # Check if the folder exists
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist!")
            continue

        # Traverse through each file in the folder
        for root, dirs, files in os.walk(folder):
            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    print(f"Processing file: {file_path}")
                    
                    # Read the file and filter out lines with the specified class_id
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    # Filter lines that do not contain the class ID to delete
                    original_line_count = len(lines)
                    filtered_lines = [line for line in lines if not line.startswith(str(class_id_to_delete))]
                    deleted_line_count = original_line_count - len(filtered_lines)

                    # Write the filtered lines back to the file
                    if deleted_line_count > 0:
                        with open(file_path, 'w') as file:
                            file.writelines(filtered_lines)
                        print(f"Deleted {deleted_line_count} labels from {file_path}")
                        total_deleted_labels += deleted_line_count
                    else:
                        print(f"No labels to delete in {file_path}")

    # Print total number of deleted labels at the end
    print(f"\nTotal number of deleted labels: {total_deleted_labels}")

if __name__ == "__main__":
    # Set the path to the folder containing 'train', 'val', and 'test' subfolders
    folder_path = "/home/ss/Kirti/lat/datasets/people_only/labels2"  # Replace with the actual folder path
    class_id_to_delete = 0  # Replace with the class ID you want to delete
    
    delete_labels_with_class_id(folder_path, class_id_to_delete)

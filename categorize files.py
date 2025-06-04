import os
import shutil

# Define the source directory where the files are located
source_dir = r"D:\GradProj\Exam Monitoring System v2.v1i.coco\valid" # Replace with your source directory

# Define target directories for good and cheat files
good_dir = r'D:\GradProj\Exam Monitoring System v2.v1i.coco\valid\good_images'  # Replace with your good files directory
cheat_dir = r"D:\GradProj\Exam Monitoring System v2.v1i.coco\valid\cheat_images"  # Replace with your cheat files directory

# Create the target directories if they don't exist
os.makedirs(good_dir, exist_ok=True)
os.makedirs(cheat_dir, exist_ok=True)

# List all files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    # Ensure we're dealing with files, not directories
    if os.path.isfile(file_path):
        # Check if the file starts with "good" or "cheat"
        if filename.lower().startswith('good'):
            # Move the file to the good directory
            shutil.move(file_path, os.path.join(good_dir, filename))
        elif filename.lower().startswith('cheat'):
            # Move the file to the cheat directory
            shutil.move(file_path, os.path.join(cheat_dir, filename))

print("Files have been categorized successfully!")

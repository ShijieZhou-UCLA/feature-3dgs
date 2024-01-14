import os
from PIL import Image

# Specify the folder path here
folder_path = 'data/flower1/input'

# List all files in the folder
files = os.listdir(folder_path)

# Filter out all JPG files
jpg_files = [file for file in files if file.lower().endswith('.jpg')]

# Convert each JPG file to PNG and delete the JPG file
for jpg_file in jpg_files:
    # Create the full file path by joining folder path and file name
    jpg_file_path = os.path.join(folder_path, jpg_file)
    
    # Open the JPG file
    with Image.open(jpg_file_path) as im:
        # Create a new file name with .png extension
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        
        # Create the full file path for the new PNG file
        png_file_path = os.path.join(folder_path, png_file)
        
        # Save the image in PNG format
        im.save(png_file_path)
    
    # Delete the original JPG file
    os.remove(jpg_file_path)
    print(f"{jpg_file} has been converted to {png_file} and the original JPG file has been deleted.")
    #print(f"{jpg_file} has been converted to {png_file}.")

print('Finished!!!')

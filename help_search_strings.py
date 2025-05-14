import os
import glob

def search_string_in_files(directory, search_string):
    # Change directory to the specified directory
    os.chdir(directory)

    # Iterate through each .py file in the directory
    for file in glob.glob("*.py"):
        with open(file, 'r') as f:
            # Read the contents of the file
            contents = f.read()
            # Check if the search string is present in the file
            if search_string in contents:
                print(f"Found '{search_string}' in file: {file}")

# Directory where .py files are located
directory = "/home/rishabh/projects/r2_gaussian"

# String to search for
search_string = ".stl"

# Call the function to search for the string in files
search_string_in_files(directory, search_string)
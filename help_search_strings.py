import os
import glob

import os
import glob

def search_string_in_files(directory, search_string):
    # Use glob to search recursively for all .py files
    py_files = glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)

    for file in py_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                contents = f.read()
                if search_string in contents:
                    print(f"Found '{search_string}' in file: {file}")
        except Exception as e:
            print(f"Could not read file {file}: {e}")


# Directory where .py files are located
directory = "/home/rishabh/projects/r2_gaussian"

# String to search for
search_string = ".npy"

# Call the function to search for the string in files
search_string_in_files(directory, search_string)
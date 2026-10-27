import os

def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not any(exclude in root for exclude in ['.git', '.venv', '__pycache__', 'node_modules']):
                print(os.path.join(root, file))

# Example usage
list_files('/path/to/your/project')

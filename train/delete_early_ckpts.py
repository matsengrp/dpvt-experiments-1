import os
import shutil

# Specify the top-level directory containing the subdirectories
top_directory = 'lightning_logs/cpu'

# Iterate over each subdirectory in the top-level directory
for subdir in os.listdir(top_directory):
    subdir_path = os.path.join(top_directory, subdir)

    # Check if the path is a directory
    if os.path.isdir(subdir_path):
        # Collect all version directories (e.g., version_0, version_1, ...)
        version_dirs = []
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)
            if os.path.isdir(item_path) and item.startswith('version_'):
                # Extract the version number from the directory name
                try:
                    version_number = int(item.split('_')[1])
                    version_dirs.append((version_number, item_path))
                except (IndexError, ValueError):
                    pass  # Ignore directories that don't match the expected pattern
        
        # Sort by version number and keep the highest one
        if version_dirs:
            version_dirs.sort()
            highest_version = version_dirs[-1]  # Keep the highest version

            # Delete all other version directories
            for version_number, version_dir_path in version_dirs:
                if version_dir_path != highest_version[1]:
                    print(f"Deleting: {version_dir_path}")
                    shutil.rmtree(version_dir_path)


            # Rename the remaining highest version directory to version_0
            new_name = os.path.join(subdir_path, 'version_0')
            print(f"Renaming {highest_version[1]} to {new_name}")
            os.rename(highest_version[1], new_name)
print("Cleanup complete.")
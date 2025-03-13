import os

# Define the directory containing the CSV files
directory = "/home/lcollien/git/dpvt-experiments-1/train/result_csvs_and_pdfs/cpu_TODAY/"
old_string = "version_1"
new_string = "version_0"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(directory, filename)

        # Read the file content
        with open(file_path, "r") as file:
            content = file.read()

        # Replace the old string with the new string
        updated_content = content.replace(old_string, new_string)

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(updated_content)

        print(f"Updated: {file_path}")

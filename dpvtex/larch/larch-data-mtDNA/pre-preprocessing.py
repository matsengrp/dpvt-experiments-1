import os
import shutil
from Bio import SeqIO

# Specify the directory containing the Nexus files
nexus_dir = '.'

# Specify the directory to save FASTA files
fasta_dir = '.'

# Iterate through each Nexus file in the directory
for filename in os.listdir(nexus_dir):
    if filename.endswith('.fasta'):
        # Extract the filename without the .nex extension
        file_name_no_ext = os.path.splitext(filename)[0]

        # Create a directory with the same name as the Nexus file (without extension)
        new_dir = os.path.join(nexus_dir, file_name_no_ext)
        os.makedirs(new_dir, exist_ok=True)

        # Move the Nexus file into the newly created directory
        src_file = os.path.join(nexus_dir, filename)
        dest_file = os.path.join(new_dir, filename)
        shutil.move(src_file, dest_file)

        # # Convert the Nexus file to FASTA format
        # fasta_file = os.path.join(fasta_dir, file_name_no_ext + '.fasta')
        # SeqIO.convert(dest_file, 'nexus', fasta_file, 'fasta')

        # print(f"Converted and moved {filename} to {fasta_file}")

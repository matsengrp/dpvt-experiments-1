import os
from pathlib import Path
from Bio import AlignIO  # Import Biopython's AlignIO for reading/writing alignment files
import sys

def convert_nexus_to_fasta(nexus_file, fasta_file):
    # Read the Nexus file and write it as FASTA
    alignment = AlignIO.read(nexus_file, "nexus")
    AlignIO.write(alignment, fasta_file, "fasta")

def process_nexus_files(src_dir):
    # List all nexus files in the source directory
    nexus_files = [f for f in os.listdir(src_dir) if f.endswith('.nex')]
    
    for nexus_file in nexus_files:
        # Create subdirectory with the same name as the nexus file (without extension)
        sub_dir_name = nexus_file.replace('.nex', '')
        sub_dir_path = os.path.join(src_dir, sub_dir_name)
        Path(sub_dir_path).mkdir(parents=True, exist_ok=True)
        
        # Define paths
        nexus_path = os.path.join(src_dir, nexus_file)
        fasta_path = os.path.join(sub_dir_path, f"{sub_dir_name}.fasta")
        
        # Convert Nexus to FASTA using Biopython
        convert_nexus_to_fasta(nexus_path, fasta_path)
        
        print(f"Processed {nexus_file} into {fasta_path}")

# Example usage
src_directory = sys.argv[1]
process_nexus_files(src_directory)


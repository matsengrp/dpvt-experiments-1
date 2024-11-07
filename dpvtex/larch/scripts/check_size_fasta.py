from Bio import SeqIO
import sys
from shutil import rmtree

def main():
    # Check whether the input fasta file has at least 5 sites
    # If it does, create a flag, so the generate_dpvt_input.snakefile is only run for data where this flag exists
    fasta_file = sys.argv[1]
    output_flag = sys.argv[2]
    for record in SeqIO.parse(fasta_file, "fasta"):
        if len(record.seq.strip()) > 5:
            with open(output_flag, 'w') as flag:
                flag.write("NOT_EMPTY")
        else:
            break


if __name__ == "__main__":
    main()

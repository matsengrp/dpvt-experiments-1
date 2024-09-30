from Bio import SeqIO
import sys
from shutil import rmtree

def main():
    fasta_file = sys.argv[1]
    output_flag = sys.argv[2]
    empty = True
    for record in SeqIO.parse(fasta_file, "fasta"):
        if len(record.seq.strip()) > 10:
            empty = False
            with open(output_flag, 'w') as flag:
                flag.write("EMPTY" if empty else "NOT_EMPTY")
        else:
            break


if __name__ == "__main__":
    main()

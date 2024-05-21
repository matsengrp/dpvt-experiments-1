from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def remove_ambiguous_characters(input_file, output_file, format='fasta'):
    # Load the alignment from file
    alignment = AlignIO.read(input_file, format)

    # Initialize a new alignment with the same sequences but empty sequences
    clean_alignment = MultipleSeqAlignment([
        SeqRecord(Seq(""), id=record.id) for record in alignment
    ])

    # Iterate over each column in the alignment
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i]
        # Include the column only if it contains no ambiguous characters
        if all(char in 'ACGT' for char in column.upper()):
            # Append this column to each sequence in the clean alignment
            for j, record in enumerate(clean_alignment):
                record.seq += Seq(column[j])

    # Save the cleaned alignment to a new file
    AlignIO.write(clean_alignment, output_file, format)
    return len(clean_alignment[0].seq) < alignment.get_alignment_length()  # Return True if any columns were removed

if __name__ == '__main__':
    import sys
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    if remove_ambiguous_characters(input_filename, output_filename):
        print("Ambiguous sites were removed.")
        sys.exit(0)
    else:
        print("No ambiguous sites found.")
        sys.exit(0)
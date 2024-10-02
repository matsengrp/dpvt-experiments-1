from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter

def remove_ambiguous_or_uninformative_sites(alignment):
    # Initialize a new alignment with the same sequences but empty sequences
    clean_alignment = MultipleSeqAlignment([
        SeqRecord(Seq(""), id=record.id.split(" ")[0]) for record in alignment
    ])
    # Iterate over each column in the alignment
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i].upper()
        if all(char in 'ACGTacgt' for char in column):
            counter = Counter(column)
            if len(counter) > 1 and not (len(counter) == 2 and 1 in counter.values()):
                # exclude uninformative sites (conserved/conserved except for one sequence)
                for j, record in enumerate(clean_alignment):                    
                    record.seq += Seq(column[j])
    return clean_alignment


def remove_duplicate_sequences(multiple_seq_alignment):
    seen_sequences = set()  # To keep track of unique sequences
    unique_alignments = []   # List to hold unique SeqRecord objects

    for record in multiple_seq_alignment:
        # Convert sequence to a string (to make it hashable)
        seq_str = str(record.seq)
        if seq_str not in seen_sequences:
            seen_sequences.add(seq_str)
            unique_alignments.append(record)
    # Create a new MultipleSeqAlignment with unique sequences
    return MultipleSeqAlignment(unique_alignments)


if __name__ == '__main__':
    import sys
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    algn_length_filename = sys.argv[3]
    alignment = AlignIO.read(input_filename, 'fasta')
    clean_alignment = remove_ambiguous_or_uninformative_sites(alignment)
    clean_alignment = remove_duplicate_sequences(clean_alignment)
    AlignIO.write(clean_alignment, output_filename, 'fasta')
    with open(algn_length_filename, "w") as f:
        f.write(str(clean_alignment.get_alignment_length()))

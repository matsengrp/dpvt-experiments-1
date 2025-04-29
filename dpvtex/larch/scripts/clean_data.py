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
    return MultipleSeqAlignment(unique_alignments), len(unique_alignments)


def trim_alignment_to_target(alignment, target_length, target_seqs):
    """
    Trims the alignment to have exactly the target number of sites and sequences, if required.
    """
    # Trim sequences if needed
    if len(alignment) > target_seqs:
        alignment = MultipleSeqAlignment(alignment[:target_seqs])
    
    # Trim alignment length if needed
    if alignment.get_alignment_length() > target_length:
        alignment = alignment[:, :target_length]
    
    return alignment


if __name__ == '__main__':
    import sys
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    algn_length_filename = sys.argv[3]
    
    # Get target dimensions if provided (optional arguments)
    target_length = None
    target_seqs = None
    if len(sys.argv) > 4:
        target_length = int(sys.argv[4])
    if len(sys.argv) > 5:
        target_seqs = int(sys.argv[5])
    else:
        print("Provide both target length and target sequences if you want to trim the alignment.")
    
    alignment = AlignIO.read(input_filename, 'fasta')
    clean_alignment = remove_ambiguous_or_uninformative_sites(alignment)
    clean_alignment, num_unique_seqs = remove_duplicate_sequences(clean_alignment)
    
    # Trim to target dimensions if specified
    if target_length is not None and target_seqs is not None:
        clean_alignment = trim_alignment_to_target(clean_alignment, target_length, target_seqs)
    
    AlignIO.write(clean_alignment, output_filename, 'fasta')
    with open(algn_length_filename, "w") as f:
        f.write(str(clean_alignment.get_alignment_length()) + "," + str(len(clean_alignment)))

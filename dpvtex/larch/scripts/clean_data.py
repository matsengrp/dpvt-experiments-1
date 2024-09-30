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
    unique_sequences = []
    seen_sequences = set()

    # Iterate over each column in the alignment
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i].upper()
        if all(char in 'ACGTacgt' for char in column):
            counter = Counter(column)
            if len(counter) > 1 and not (len(counter) == 2 and 1 in counter.values()):
                # exclude uninformative sites (conserved/conserved except for one sequence)
                for j, record in enumerate(clean_alignment):                    
                    record.seq += Seq(column[j])
                    new_sequence = record.seq
                    # Check if the new sequence is already added
                    if str(new_sequence) not in seen_sequences:
                        seen_sequences.add(str(new_sequence))
                        record.name = ""
                        record.description = ""
                        unique_sequences.append(record)
    output_alignment = MultipleSeqAlignment(unique_sequences)
    return output_alignment


if __name__ == '__main__':
    import sys
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    algn_length_filename = sys.argv[3]
    alignment = AlignIO.read(input_filename, 'fasta')
    clean_alignment = remove_ambiguous_or_uninformative_sites(alignment)
    AlignIO.write(clean_alignment, output_filename, 'fasta')
    with open(algn_length_filename, "w") as f:
        f.write(str(clean_alignment.get_alignment_length()))

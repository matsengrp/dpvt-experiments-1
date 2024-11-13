from dpvtex.larch.scripts.clean_data import remove_ambiguous_or_uninformative_sites

from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq



def are_alignments_equal(align1, align2):
    # Check if two alignments are equal.
    # This assumes that the order of sequences is the same in both alignments
    if len(align1) != len(align2) or align1.get_alignment_length() != align2.get_alignment_length():
        return False
    for seq_record1, seq_record2 in zip(align1, align2):
        # Compare identifiers and sequences
        if (seq_record1.id != seq_record2.id) or (str(seq_record1.seq) != str(seq_record2.seq)):
            return False
    return True



def test_remove_ambiguous_or_uninformative_sites():
    seq1 = SeqRecord(Seq("ACTGCCAGCT-G"), id="Seq1")
    seq2 = SeqRecord(Seq("ACAGACAGCATG"), id="Seq2")
    seq3 = SeqRecord(Seq("ACTCTTAGCCCG"), id="Seq3")
    seq4 = SeqRecord(Seq("ACT-CTAGCGA-"), id="Seq4")
    alignment = MultipleSeqAlignment([seq1, seq2, seq3, seq4])

    expected_seq1 = SeqRecord(Seq("CCT"), id="Seq1")
    expected_seq2 = SeqRecord(Seq("ACA"), id="Seq2")
    expected_seq3 = SeqRecord(Seq("TTC"), id="Seq3")
    expected_seq4 = SeqRecord(Seq("CTG"), id="Seq4")

    expected_alignment = MultipleSeqAlignment([expected_seq1, expected_seq2, expected_seq3, expected_seq4])

    cleaned_alignment = remove_ambiguous_or_uninformative_sites(alignment)

    assert are_alignments_equal(expected_alignment, cleaned_alignment)
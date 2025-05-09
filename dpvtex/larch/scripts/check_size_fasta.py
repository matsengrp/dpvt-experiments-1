import sys
from shutil import rmtree

def main():
    # Check whether the input fasta file has at least 5 sites If it does, create
    # a flag, so the generate_dpvt_input.snakefile is only run for data where
    # this flag exists
    algn_length = sys.argv[1]
    output_flag = sys.argv[2]
    with open(algn_length, "r") as f:
        [msa_length, msa_size] = f.readline().split(",")
    if int(msa_length) >= 5 and int(msa_size) >= 5:
        with open(output_flag, 'w') as flag:
            flag.write("NOT_EMPTY")
    else:
        data_name = algn_length.split("/")[-2]
        print("Alignment " + data_name + " contains less than five sequences or less than five sites after removing uninformative sites. Skip this dataset.")


if __name__ == "__main__":
    main()

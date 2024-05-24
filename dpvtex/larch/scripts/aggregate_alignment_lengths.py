import pandas as pd

input_files = snakemake.input[0]
output_file = snakemake.output[0]

length_list = []
for file in input_files:
    with open(file, "r") as f:
        length_list.append(int(f.readline().strip()))

df = pd.DataFrame({"algn_lengths": length_list})
pd.to_csv(df, output_file)
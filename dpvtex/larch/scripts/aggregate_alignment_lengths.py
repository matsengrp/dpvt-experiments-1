import pandas as pd

input_files = snakemake.input.length_files
output_file = snakemake.output.all_algn_lengths

length_list = []
for file in input_files:
    with open(file, "r") as f:
        length_list.append(int(f.readline().strip()))

df = pd.DataFrame({"algn_lengths": length_list})
df.to_csv(output_file)

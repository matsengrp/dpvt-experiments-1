import pandas as pd
import sys

# Read the log file into a DataFrame
log_file = sys.argv[1]  # Assumes log file path is passed as argument
df = pd.read_csv(log_file, sep='\t')

# print("logfile: ", log_file)

num_rows = 10

# Get the last num_rows rows of the DataFrame
last_rows = df.tail(num_rows)

# print(last_rows)
# Check if MaxParsimony remains the same in the last 5 rows
max_parsi_last_rows = last_rows['MaxParsimony'].values
# print(max_parsi_last_rows)
if len(set(max_parsi_last_rows)) == 1:
    # print(f"MaxParsimony remains the same in the last {num_rows} rows.")
    sys.exit(0)  # Exit with code 0 (no change)
else:
    # print(f"MaxParsimony changed in the last {num_rows} rows. Rerun the program.")
    sys.exit(1)  # Exit with code 1 (change detected)

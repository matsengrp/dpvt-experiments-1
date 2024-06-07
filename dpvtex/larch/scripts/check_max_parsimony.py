import pandas as pd
import sys

# Read the log file into a DataFrame
log_file = sys.argv[1]  # Assumes log file path is passed as argument
df = pd.read_csv(log_file, sep='\t')

# Get the last 5 rows of the DataFrame
last_5_rows = df.tail(5)

# Check if MaxParsimony remains the same in the last 5 rows
max_parsi_last_5 = last_5_rows['MaxParsimony'].values
if len(set(max_parsi_last_5)) == 1:
    print("MaxParsimony remains the same in the last 5 rows.")
    sys.exit(0)  # Exit with code 0 (no change)
else:
    print("MaxParsimony changed in the last 5 rows. Rerun the program.")
    sys.exit(1)  # Exit with code 1 (change detected)

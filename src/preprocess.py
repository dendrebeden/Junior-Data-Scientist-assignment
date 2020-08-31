import pandas as pd
import numpy as np

def preprocess(input_file, output_file):
    """Removes 'Name', 'Ticket' and 'Cabin' columns from input file

    Args:
        input_file : string
            csv input file name with ';' separator
        output_file : string
            csv output file name
    """
    # Import data from input_file.
    df = pd.read_csv(input_file, sep = ";")
    assert(all([item in df.columns for item in ["Name", "Ticket", "Cabin", "PassengerId"]]))
    df = df.dropna()

    # Drop columns where features are making no sense.
    df.drop(["Name", "Ticket", "Cabin", "PassengerId"], inplace = True, axis = 1)

    # Export data to output_file.
    df.to_csv(output_file, index = False)
import pandas as pd
import numpy as np

def age_misssing(cols, age_dict):
    Pclass = cols[0]
    Age = cols[1]
    if pd.isnull(Age):
        return age_dict[Pclass]
    else:
        return Age

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
    # Substitute 'Age' column values with mean by 'Pclass'.
    age_mean_by_pclass = df.groupby(by=["Pclass"]).mean()["Age"]
    
    df["Age"] = df[["Pclass", "Age"]].apply(lambda x: age_misssing(x, age_mean_by_pclass), axis = 1)

    # Drop columns where features are making no sense.
    df.drop(["Name", "Ticket", "Cabin", "PassengerId"], inplace = True, axis = 1)
    df.dropna(inplace = True)
    
    # Export data to output_file.
    df.to_csv(output_file, index = False)
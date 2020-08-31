import pandas as pd

def build_features(input_file, output_file):
    """Function substitutes 'Sex' and 'Embarked' columns values with dummies and creates
       'FamilySize' and 'IsAlone' columns

    Args:
        input_file : string
            csv input file name
        output_file : string
            csv output file name
    """
    # Import data from input_file.
    df = pd.read_csv(input_file)
    assert(all([item in df.columns for item in ["Sex", "Embarked", "SibSp", "Parch"]]))

    # Substitute 'Sex' column values with dummies.
    df["Sex"] = df["Sex"].replace("male", 0)
    df["Sex"] = df["Sex"].replace("female", 1)
    
    # Substitute 'Embarked' column values with dummies.
    embarked_dict = {}
    embarked_dict_values = 0
    for i in df.Embarked.unique():
        embarked_dict_values = embarked_dict_values + 1
        embarked_dict[i] = embarked_dict_values

    for i in embarked_dict.keys():
        df["Embarked"].replace(i, embarked_dict[i], inplace = True)

    # Calculate family size column.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # Define alone passengers
    df["IsAlone"] = df["FamilySize"].apply(lambda x: 1 if x == 1 else 0)
    
    # Export data to output_file.
    df.to_csv(output_file, index = False)

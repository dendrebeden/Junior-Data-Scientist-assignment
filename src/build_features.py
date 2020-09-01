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
    assert(all([item in df.columns for item in ["Sex", "Embarked"]]))

    # Substitute 'Sex' and 'Embarked' columns values with dummies.
    sex = pd.get_dummies(df['Sex'], drop_first=True)
    embark = pd.get_dummies(df['Embarked'], drop_first=True)
    df = pd.concat([df,sex,embark], axis=1)
    df.drop(['Sex','Embarked'], axis=1, inplace=True)

    # Export data to output_file.
    df.to_csv(output_file, index = False)
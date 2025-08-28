import pandas as pd

data = pd.read_csv(r"./EstData.csv")


if __name__ == "__main__":

    for col in data.columns:
        print(f"Column: {col}")
        print(data[col].value_counts(dropna=False))
        print("-" * 40)

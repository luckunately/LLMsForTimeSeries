import sys
import pandas as pd

def analyze_csv(file_path):
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Analyze each column
        for column in data.columns:
            # For categorical columns
            most_common = data[column].value_counts().idxmax()
            count = data[column].value_counts().max()
            percentage = (count / len(data[column])) * 100
            print(f"Column '{column}': Most occurred value is '{most_common}' with {percentage:.2f}% occurrence.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_code.py <csv_file_path>")
    else:
        analyze_csv(sys.argv[1])
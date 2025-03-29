import pandas as pd
import sys

def process_page_fault(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Generate a date range starting from 2016-01-01
    date_range = pd.date_range(start='2016-01-01', periods=len(df), freq='D')
    
    # Insert the date range as a new column at the beginning of the dataframe
    df.insert(0, 'date', date_range)
    
    # Cleaned dataframe
    df_cleaned = df
    
    # Write the processed dataframe to the output CSV file
    df_cleaned.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_process_page_fault.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    process_page_fault(input_csv, output_csv)
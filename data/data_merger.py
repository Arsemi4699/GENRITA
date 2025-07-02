import pandas as pd

def merge_csv_files(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    merged_df = pd.concat([df1, df2], ignore_index=True)

    merged_df.to_csv(output_path, index=False)
    return merged_df

# Example usage
if __name__ == "__main__":
    file1 = 'cleaned_data_merged.csv'
    file2 = 'cleaned_data3.csv'
    output = 'cleaned_data_merged.csv'

    merged_data = merge_csv_files(file1, file2, output)
    print("Merge completed. Result shape:", merged_data.shape)

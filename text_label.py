import pandas as pd

# Read the existing CSV file
df = pd.read_csv('./iemocap.csv')

# Select only the "Sentences" and "Label" columns
new_df = df[['Sentences', 'Label']].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Map the 'Label' column values to numbers
label_mapping = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3, 'exc': 4}
new_df['Label_num'] = new_df['Label'].map(label_mapping)


# Save the new DataFrame to a new CSV file
new_df.to_csv('./text_label_file_iemocap.csv', index=False)
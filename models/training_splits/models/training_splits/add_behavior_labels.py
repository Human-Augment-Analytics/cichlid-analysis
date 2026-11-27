import pandas as pd

# Load the Parquet file
df = pd.read_parquet('../../data/0028_vid.parquet')

# Load the annotations CSV
annotations = pd.read_csv('../../ground_truth_annotations/finalized_annotations/circling_annotations_vid0028.csv')

# Initialize the Behavior column with 'None'
df['Behavior'] = 'None'

# Ensure the frame columns are numeric
annotations['Start Frame'] = pd.to_numeric(annotations['Start Frame'], errors='coerce')
annotations['End Frame'] = pd.to_numeric(annotations['End Frame'], errors='coerce')

# Drop potential NaNs if any rows are malformed
annotations = annotations.dropna(subset=['Start Frame', 'End Frame'])

# Iterate over the annotation rows and update the Behavior column
for _, row in annotations.iterrows():
    start = row['Start Frame']
    end = row['End Frame']
    df.loc[(df['FrameNum'] >= start) & (df['FrameNum'] <= end), 'Behavior'] = 'Circling'

# Save the resulting DataFrame to Parquet
df.to_parquet('../../ground_truth_annotations/finalized_annotations/0028_vid_labeled.parquet', index=False)
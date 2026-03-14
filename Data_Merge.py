import os
import pandas as pd

# 1. Your main local folder containing all the dataset folders
base_dir = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs"

print(f"🕵️‍♂️ Scanning {base_dir} for ALL dataset metadata...\n")

all_dataframes = []

# 2. Walk through every folder looking for the mini-CSVs
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith("_metadata.csv") and "MASTER" not in file:
            csv_path = os.path.join(root, file)
            
            try:
                # Try to read the CSV
                df = pd.read_csv(csv_path)
                
                # If it's empty but still has headers, it might have 0 rows
                if len(df) == 0:
                    print(f"⚠️ Skipping {file}: File is empty (0 rows).")
                    continue
                    
                print(f"✅ Loaded {len(df)} rows from: {file}")
                
                # 🔧 THE MAGIC FIX: Replace cloud paths with local paths
                df['file_path'] = df['file_path'].apply(lambda x: os.path.join(root, os.path.basename(x)))
                
                all_dataframes.append(df)
                
            except pd.errors.EmptyDataError:
                print(f"🚨 ERROR: {file} is completely blank/corrupted! Skipping it.")
            except Exception as e:
                print(f"🚨 ERROR reading {file}: {e}")

# 3. Combine them all into the True Master CSV
if len(all_dataframes) > 0:
    super_df = pd.concat(all_dataframes, ignore_index=True)
    
    super_csv_path = os.path.join(base_dir, "TRUE_MASTER_VOXALIGN_DATASET.csv")
    super_df.to_csv(super_csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"🚀 TRUE MASTER DATASET CREATED: {len(super_df)} Total Audio Tensors!")
    print("="*50)
    
    # Check the ultimate class balance!
    emotion_names = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fear', 5: 'Disgust', 6: 'Surprise'}
    super_df['Emotion_Name'] = super_df['emotion_id'].map(emotion_names)
    
    print("\n📊 ULTIMATE Emotion Class Balance:")
    print(super_df['Emotion_Name'].value_counts())
    print(f"\n📁 Saved to: {super_csv_path}")
else:
    print("\n🚨 All CSVs were empty or missing!")
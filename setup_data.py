import os
import re
import pandas as pd

"""
This script takes the images and puts them into a dataframe encoding 'Infestation' attribute as 1 or 0.
Comments have been added using LLMs. Let me know if there are any issues/unclarities.
"""

# Define the parent directory where all index-type folders are stored
parent_dir = "data"

# Initialize an empty list to store data
data_infested = []
data_non_infested = []

# Loop through each folder inside the parent directory
for folder in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder)

    # Check if it is a directory (not a file)
    if os.path.isdir(folder_path):
        # Determine the index type and infestation status from folder name
        match = re.match(r"(CIG|EVI|NDVI)_(infected|non-infected)", folder, re.IGNORECASE)

        if match:
            index_type, infestation_status = match.groups()

            infestation_label = 1 if infestation_status.lower() == "infected" else 0  # Convert to binary

            # Loop through each image file in the folder
            for file_name in os.listdir(folder_path):
                # Extract metadata using regex
                match = re.match(r"(\d{4}-\d{2}-\d{2})_(cig|evi|ndvi)_(infected|non-infected) \((\d+)\)", file_name, re.IGNORECASE)
                if match:
                    date, file_index_type, file_status, field_id = match.groups()

                    # Ensure consistency in case handling
                    file_index_type = file_index_type.upper()
                    if file_status == 'infected':
                        field_id = str(int(field_id) + 11)
                        # Append infested data to the list
                        data_infested.append([date, field_id, file_index_type, infestation_label, os.path.join(folder_path, file_name)])
                    if file_status == 'non-infected':
                        data_non_infested.append([date, field_id, file_index_type, infestation_label, os.path.join(folder_path, file_name)])


# Concatenate infested and non-infested lists
data = data_infested + data_non_infested


# to pandas dataframe
df = pd.DataFrame(data, columns=["Date", "Field_ID", "Index_Type", "Infestation", "File_Path"])


# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])


# Pivot data to have CIG, EVI, NDVI as columns
df_pivot = df.pivot_table(index=["Date", "Field_ID"], columns="Index_Type", values="File_Path", aggfunc="first").reset_index()


# Rename columns for clarity
df_pivot.columns.name = None  # Remove column name groupings
df_pivot.rename(columns={"CIG": "CIG_Path", "EVI": "EVI_Path", "NDVI": "NDVI_Path"}, inplace=True)


# Merge with infestation labels
df_final = df_pivot.merge(df[['Date', 'Field_ID', 'Infestation']].drop_duplicates(), on=['Date', 'Field_ID'], how='left')


# Fill missing values (in case some fields may lack certain indices)
df_final.fillna("", inplace=True)


# Export data frame to a csv file
df_final.to_csv("farm_data.csv", index=False)

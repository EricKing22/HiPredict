"""
File to download and parse metadata files from HiRISE PDS.
Also to generate EDR dataframe from these files.
"""


import requests
import pandas as pd
import re
import os

from hipredict.paths import files_dir

METADATA_DIR = files_dir() / "metadata"


def download_file(url, local_filename):
    headers = {}
    
    # Check if the file already exists and get its size
    if os.path.exists(local_filename):
        file_size = os.path.getsize(local_filename)
        return local_filename
    
    file_size = 0
    print(f"{local_filename} not found. Downloading...")
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    # Download to a temporary name so an interrupted download is never mistaken for the complete file
    partial_filename = f"{local_filename}.part"

    try:
        # Timeouts (connect, read) in seconds, so a stalled server raises instead of hanging
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            # Check if the server supports range requests and if the file is fully downloaded
            if response.status_code == 416:
                print("File is already fully downloaded.")
                return local_filename
            
            response.raise_for_status()
            
            # Check if partial content is supported
            if "Content-Range" in response.headers:
                total_size = int(response.headers["Content-Range"].split("/")[-1])
                if file_size >= total_size:
                    print("EDR data file is already fully downloaded.")
                    return local_filename
            else:
                # If server does not support range requests, reset file size to zero
                file_size = 0

            # Set file write mode and start downloading
            mode = "ab" if file_size > 0 else "wb"
            with open(partial_filename, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        os.replace(partial_filename, local_filename)
        print("Download completed successfully.")
        return local_filename

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return None


def parse_tab_file(url):
    local_filename = str(METADATA_DIR / "metadata_file.tab")
    try:
        file_name = download_file(url, local_filename)
        df = pd.read_csv(file_name, header=None, delimiter=",")

        str_cols = df.select_dtypes(include=['object']).columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.replace(" ", "").str.strip('"'))

        object_data = df.values.tolist()
        return object_data

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the .tab file: {e}")
        return None


def parse_lbl_file(url):
    local_filename = str(METADATA_DIR / "metadata_file.lbl")
    local_filename = download_file(url, local_filename)

    if local_filename is None:
        print("Error downloading the .lbl file.")
        return None

    with open(local_filename, "r") as f:
        lines = f.readlines()

    columns_data = []
    in_column_object = False
    column_metadata = {}

    for line in lines:
        line = line.strip()

        # Start of a COLUMN object
        if line.startswith("OBJECT") and "COLUMN" in line:
            in_column_object = True
            column_metadata = {}

        # End of a COLUMN object
        elif line.startswith("END_OBJECT") and in_column_object:
            columns_data.append(column_metadata)  # Save the parsed column metadata
            in_column_object = False

        # Parse key-value pairs within a COLUMN object
        elif "=" in line and in_column_object:
            key, value = map(str.strip, line.split("=", 1))

            # Clean up value format (remove quotes, convert to int if needed)
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.isdigit():
                value = int(value)
            elif re.match(r'^\d+\.\d+$', value):
                value = float(value)

            if "ITEMS" in key:
                item_count = int(value)
                column_metadata["ITEMS"] = item_count
            else:
                column_metadata[key] = value

    # Correct for columns with multiple items
    expanded_col_data = []
    for col_metadata in columns_data:
        if "ITEMS" in col_metadata:
            item_count = col_metadata.pop("ITEMS")
            base_metadata = col_metadata.copy()

            for i in range(1, item_count + 1):
                updated_metadata = base_metadata.copy()
                updated_metadata["NAME"] = f"{base_metadata['NAME']}_{i}"
                expanded_col_data.append(updated_metadata.copy())

        else:
            expanded_col_data.append(col_metadata)

    df = pd.DataFrame(expanded_col_data)
    return df


def gen_edr_df(EDR_csv_path=str(METADATA_DIR / 'EDR_df.csv'), verbose=False):
    if os.path.exists(EDR_csv_path):
        EDR_df = pd.read_csv(EDR_csv_path)
        if verbose:
            print(f"Loaded EDR df with {len(EDR_df):,} images and {EDR_df['OBSERVATION_ID'].nunique():,} unique observations.")
        return EDR_df

    print("Warning: EDR df file not found. Downloading and generating EDR df from scratch. This may take a few minutes...")
    EDR_lbl_url = 'https://hirise-pds.lpl.arizona.edu/PDS/INDEX/EDRCUMINDEX.LBL'
    EDR_lbl_df = parse_lbl_file(EDR_lbl_url)
    assert EDR_lbl_df is not None, "Failed to parse EDR .LBL file. Cannot generate EDR df."
    col_names = EDR_lbl_df['NAME']

    EDR_tab_url = 'https://hirise-pds.lpl.arizona.edu/PDS/INDEX/EDRCUMINDEX.TAB'
    EDR_tab_file = parse_tab_file(EDR_tab_url)

    EDR_df = pd.DataFrame(EDR_tab_file, columns=col_names)

    if verbose:
        print(f"Generated EDR df with {len(EDR_df):,} images and {EDR_df['OBSERVATION_ID'].nunique():,} unique observations.")

    EDR_df = EDR_df[EDR_df['IMAGE_LINES'] != 0] # Remove entries with 0 image lines

    required_ccds = ['RED3', 'RED4', 'RED5', 'IR10', 'BG12']
    valid_obs = EDR_df.groupby('OBSERVATION_ID')['CCD_NAME'].apply(
        lambda x: set(required_ccds).issubset(set(x.unique()))
    ).sum()

    full_ccds = ['RED0', 'RED1', 'RED2', 'RED3', 'RED4', 'RED5', 'RED6', 'RED7', 'RED8', 'IR10', 'IR11', 'BG12', 'BG13']
    full_obs = EDR_df.groupby('OBSERVATION_ID')['CCD_NAME'].apply(
        lambda x: set(full_ccds).issubset(set(x.unique()))
    ).sum()       

    if verbose:
        print(f"Observations with 5 key CCDs:              {valid_obs:,}")
        print(f"Observations with all 13 CCDs (ex R9):     {full_obs:,}\n")

    EDR_df.to_csv(EDR_csv_path, index=False)

    # File size
    if verbose:
        print(f"Saved EDR df file size: {os.path.getsize(EDR_csv_path) / 1e6:.2f} MB at {EDR_csv_path}")

    return EDR_df

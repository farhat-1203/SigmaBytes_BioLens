import pandas as pd
import urllib.request
import gzip
import re
url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50760/matrix/GSE50760_series_matrix.txt.gz"
urllib.request.urlretrieve(url, "GSE50760_series_matrix.txt.gz")

def parse_geo_matrix(filename):
    metadata = {}
    sample_info = {}
    with gzip.open(filename, 'rt') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('!series_matrix_table_begin'):
            break
        if line.startswith('!Sample_'):
            parts = line.split('\t')
            key = parts[0].replace('!Sample_', '').replace('"', '')
            values = [val.strip('"') for val in parts[1:]]
            sample_info[key] = values
        elif line.startswith('!Series_'):
            parts = line.split('\t', 1)
            key = parts[0].replace('!Series_', '').replace('"', '')
            value = parts[1].strip('"') if len(parts) > 1 else ''
            metadata[key] = value

    return metadata, sample_info

def create_sample_dataframe(sample_info):
    max_samples = max(len(values) for values in sample_info.values()) if sample_info else 0
    df_data = {}
    for key, values in sample_info.items():
    
        padded_values = values + [None] * (max_samples - len(values))
        df_data[key] = padded_values

    df = pd.DataFrame(df_data)
    return df
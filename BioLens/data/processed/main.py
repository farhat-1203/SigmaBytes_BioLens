import GEOparse
import pandas as pd

geo_id = "GSEXXXXX"
gse = GEOparse.get_GEO(geo=geo_id)

dfs = []
for gsm_name, gsm in gse.gsms.items():
    df = gsm.table[['ID_REF', 'VALUE']].copy()
    df.columns = ['probe_id', gsm_name]
    dfs.append(df)
expression_df = dfs[0].merge(dfs[1], on='probe_id')
for df in dfs[2:]:
    expression_df = expression_df.merge(df, on='probe_id')

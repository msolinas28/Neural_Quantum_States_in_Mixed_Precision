import pandas as pd
import os

def create_or_append(data, folder, name):
    out = folder + name
    os.makedirs(folder, exist_ok=True)
    df_new = pd.DataFrame(data)
    exist = os.path.isfile(out)

    if exist:
        df_existing = pd.read_csv(out)
        all_columns = sorted(set(df_existing.columns).union(df_new.columns))
        df_existing = df_existing.reindex(columns=all_columns)
        df_new = df_new.reindex(columns=all_columns)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(out, index=False)
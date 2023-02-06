# Notes
'''
Author: Gyumin Lee
Version: 0.1
Description (primary changes): Collect claims for the patents in "collection_final.csv"
'''
# Set root directory
root_dir = '/home2/glee/Tech_Gen/'

import sys
sys.path.append("/share/uspto_pkg")
import os
import pandas as pd
from tqdm import tqdm
import uspto

rawdata = pd.read_csv(os.path.join(root_dir,"data","collection_final.csv"))
pns = rawdata['number'].apply(lambda x: str(x))

uspto.shared.set_html_dir("/share/uspto/html")

claims = []
pns_with_claims = []
for pn in tqdm(pns):
    try:
        p = uspto.Patent(pn)
        claims.append(p.claims)
        pns_with_claims.append(int(pn))
    except:
        continue

newdata = rawdata.set_index('number').loc[pns_with_claims].reset_index()
newdata = pd.concat([newdata, pd.Series(claims).rename('claims')], axis=1)

newdata.to_csv(os.path.join(root_dir,"data","collection_final_with_claims.csv"))

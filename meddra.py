# Import pandas
import pandas as pd
from tqdm import tqdm
# reading csv file
df = pd.read_csv(r'C:\Users\lu\Desktop\MEDDRA_raw.csv')
df
llt_df = df[['llt_code','llt_name_en','llt_name']]
pt_df = df[['pt_code','pt_name_en','pt_name']]
hlt_df = df[['hlt_code','hlt_name_en','hlt_name']]
hlgt_df = df[['hlgt_code','hlgt_name_en','hlgt_name']]
soc_df = df[['soc_code','soc_name_en','soc_name']]
d  =dict()
i = 0
for llt,pt,hlt,hlgt,soc in tqdm(zip(llt_df.iterrows(),pt_df.iterrows(),hlt_df.iterrows(),hlgt_df.iterrows(),soc_df.iterrows())):
    dd =d
    soc_t = tuple(soc)
    soc_t = tuple(soc_t[1].items())
    if soc_t not in dd:
        dd[soc_t] = dict()
    dd = dd[soc_t]

    hlgt_t = tuple(hlgt)
    hlgt_t = tuple(hlgt_t[1].items())
    if hlgt_t not in dd:
        dd[hlgt_t] = dict()
    dd = dd[hlgt_t]

    hlt_t = tuple(hlt)
    hlt_t = tuple(hlt_t[1].items())
    if hlt_t not in dd:
        dd[hlt_t] = dict()
    dd = dd[hlt_t]

    pt_t = tuple(pt)
    pt_t = tuple(pt_t[1].items())
    if pt_t not in dd:
        dd[pt_t] = dict()
    dd = dd[pt_t]

    llt_t = tuple(llt)
    llt_t = tuple(llt_t[1].items())
    if llt_t not in dd:
        dd[llt_t] = dict()
    dd = dd[llt_t]
    i+=1
decodedd = []

def camelCase(st):
    output = ''.join(x for x in st.title() if x.isalnum())
    return output[0].upper() + output[1:]

def decode(decoded:list, node: dict):

    for k,v in node.items():
        ddd = {}
        decoded.append(ddd)
        for kk in k:
            ddd[camelCase("_".join(kk[0].split('_')[1:]))] = kk[1]
        if not (len(v) == 0):
            ddd['Children'] = []
            decode(ddd['Children'],v)

decode(decodedd,d)
import json

# Serializing json
json_object = json.dumps(decodedd, indent=4, ensure_ascii=False)

# Writing to sample.json
with open("meddra.json", "w",encoding='utf-8') as outfile:
    outfile.write(json_object)
datalist = [
    '../molport/train.csv',
    '../molport/val.csv'
]

dic = {}
with open('library_used.csv') as f :
    lines = f.readlines()
    for l in lines[1:] :
        fid, _, _, org_fid = l.strip().split(',')
        dic[int(org_fid)] = fid

with open('train.csv', 'w') as w :
    w.write('SMILES,FID,Idx,MolID\n')
    with open('../molport/train.csv') as f :
        lines = f.readlines()
        for l in lines[1:] :
            smi, org_fid, idx, molid = l.split(',')
            fid = dic.get(int(org_fid), None)
            if fid is not None :
                w.write(f'{smi},{fid},{idx},{molid}')

with open('val.csv', 'w') as w :
    w.write('SMILES,FID,Idx,MolID\n')
    with open('../molport/val.csv') as f :
        lines = f.readlines()
        for l in lines[1:] :
            smi, org_fid, idx, molid = l.split(',')
            fid = dic.get(int(org_fid), None)
            if fid is not None :
                w.write(f'{smi},{fid},{idx},{molid}')

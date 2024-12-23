import os,sys
import json

with open('data_nicknames.json', 'r') as fp:
    data_nicknames = json.load(fp)

print(f'data_nicknames:\n{data_nicknames}')

data_dir = data_nicknames["data_dir"]
for nickname,filename in data_nicknames.items():
    if nickname == 'data_dir':
        continue
    path = f'{data_dir}/{filename}'
    file_exists = 'EXISTS' if os.path.exists(path) else 'DOES_NOT_EXIST'
    print(f'{path}: {file_exists}')

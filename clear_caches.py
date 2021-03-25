import os
import yaml
import shutil

with open('config.yaml', 'r', encoding='utf-8')as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

for folder in cfg['FOLDER']:
    shutil.rmtree(folder, True)
    print('Remove dir: {} ok.'.format(folder))
    shutil.rmtree(os.path.join('static', folder), True)
    print('Remove dir: {} ok.'.format(os.path.join('static', folder)))

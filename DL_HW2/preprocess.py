import os
import sys
import glob
from PIL import Image


'''
Preprocess Image
resize to 256x256 by high quality antialias
'''

'''
# only use with jupyter notebook in vscode
if 'DL_HW2' not in os.getcwd():
    os.chdir(os.getcwd()+'/DL_HW2')
'''

data_path = 'data/animal-10/'
exts = ['.png', '.jpg', '.jpeg']
paths = []
for ext in exts:
    path = os.path.join(data_path, '**', '*'+ext)
    res = [p for p in glob.glob(path, recursive=True)]
    paths.extend(res)

save_path = 'data/processed/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i, p in enumerate(paths):
    image = Image.open(p)
    new_img = image.resize((256, 256), Image.ANTIALIAS)

    data_path = os.path.normpath(p).split(os.sep)
    dir_path = os.path.join(save_path, data_path[-3], data_path[-2])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    new_img.save(os.path.join(dir_path, data_path[-1]))
    sys.stdout.write('\r{}/{}'.format(i+1, len(paths)))
print()

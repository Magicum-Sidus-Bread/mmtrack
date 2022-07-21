import os
import shutil

path = 'G:/大创项目/test_process_data/train'
new_path = 'G:/大创项目/test_process_data/new_train'
count = os.listdir(path)
print(count)
num = 0
for j in range(1,len(count)+1):
    for root, dirs, files in os.walk(path):
        if len(dirs) == 0:
            for file in files:
                if file.find('.jpg') != -1:
                    shutil.copy(os.path.join(path +'/'+ str(j)+'/'+ 'img', file),os.path.join(new_path, 'P'+ str(num).zfill(4) + '.jpg'))
                    num = num + 1



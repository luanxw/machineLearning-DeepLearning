import os

def search_flies(directory):
    '''
    检索 directory 目录下所有的jpg文件返回字典目录
    '''
    objects = {}
    for curdir ,subdir , files in  os.walk(directory):
        for file in files:
            if file.endswith('jpg'):
                label = curdir.split(os.path.sep)[-1]
                if label not in objects:
                    objects[label] = []
                url = os.path.join(curdir, file)
                objects[label].append(url)
    return objects
data = search_flies('../machineLearning/data/computer')
print(data)

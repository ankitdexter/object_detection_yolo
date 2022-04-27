import os, shutil, random

source = os.listdir('/home/dexter/Projects/interview/eagleView/data/trainval/annotations')

#splitting on the basis of random function with 6:2:2 ratio on train test and validation data
split = int(len(source)*0.2)
lii = random.sample(source,int(len(source)*0.2))
for fil in lii:
    shutil.move(f'/home/dexter/Projects/interview/eagleView/data/trainval/annotations/{fil}',f'/home/dexter/Projects/interview/eagleView/data/trainval/validation/annotations/{fil}')
    shutil.move(f'/home/dexter/Projects/interview/eagleView/data/trainval/images/{fil[:-3]}jpg',f'/home/dexter/Projects/interview/eagleView/data/trainval/validation/images/{fil[:-3]}jpg')

source = list(set(source) - set(lii))
lii = random.sample(source,split)
for fil in lii:
    shutil.move(f'/home/dexter/Projects/interview/eagleView/data/trainval/annotations/{fil}',f'/home/dexter/Projects/interview/eagleView/data/trainval/test/annotations/{fil}')
    shutil.move(f'/home/dexter/Projects/interview/eagleView/data/trainval/images/{fil[:-3]}jpg',f'/home/dexter/Projects/interview/eagleView/data/trainval/test/images/{fil[:-3]}jpg')

source = list(set(source) - set(lii)) 
for fil in source:
    shutil.move(f'/home/dexter/Projects/interview/eagleView/data/trainval/annotations/{fil}',f'/home/dexter/Projects/interview/eagleView/data/trainval/train/annotations/{fil}')
    shutil.move(f'/home/dexter/Projects/interview/eagleView/data/trainval/images/{fil[:-3]}jpg',f'/home/dexter/Projects/interview/eagleView/data/trainval/train/images/{fil[:-3]}jpg')

print('done')
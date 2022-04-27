import json
from tqdm import tqdm

def writeAnnotFile(annot_dict, save_path = 'data/trainval/annotations/'):
    '''
    This function writes annotation into XML format (PASCAL-VOC)
    '''
    filename = annot_dict['file_name']
    shape = [annot_dict['width'], annot_dict['height']]
    
    objs = ''''''
    for obj in annot_dict['objects']:
        label = 'person' if obj['category_id'] is 1 else 'car'
        box = obj['bbox']
        objs+=f'''
    <object>
        <name>{label}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{int(box[0])}</xmin>
            <ymin>{int(box[1])}</ymin>
            <xmax>{int(box[0])+int(box[2])}</xmax>
            <ymax>{int(box[1])+int(box[3])}</ymax>
        </bndbox>
    </object>
                '''

    finalXML = f'''
<annotation>
    <folder>images</folder>
    <filename>{filename}</filename>
    <path>/home/dexter/Projects/interview/eagleView/data/trainval/images/{filename}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{shape[1]}</width>
        <height>{shape[0]}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
{objs}
</annotation>
    '''
    with open(save_path + filename[:-3] + 'xml','w') as f:
        f.write(finalXML)
    return True

if __name__ == '__main__':
    annot_file_path = 'data/trainval/annotations/bbox-annotations.json'
    #loading json
    with open(annot_file_path) as f:
        annot_dict = json.load(f)

    #moving all information to images key
    for i, ann_dic in enumerate(annot_dict['annotations']):
        idd = ann_dic['image_id']
        if 'objects' not in annot_dict['images'][idd]:
            annot_dict['images'][idd]['objects'] = [ann_dic]
        else:
            annot_dict['images'][idd]['objects'].append(ann_dic)

    #writing files to trainval folder path
    for dic in tqdm(annot_dict['images']):
        writeAnnotFile(dic)
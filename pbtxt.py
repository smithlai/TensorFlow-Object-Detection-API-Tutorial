import argparse
import os.path
DEFAULT_IMAGE_ROOT = 'images'
def read_tags(source_root):
    with open(os.path.join(source_root,'predefined_classes.txt')) as f:
        lines = f.readlines()
    tag_num = 0;
    tags = []
    for tag in lines:
        tag = tag.strip()
        if tag:
            tag_num += 1
            print (str(tag_num) + " " + tag)
            tags.append(tag)
    return tags;

def read_temp1(tags, source_root):
    a = "    if row_label == '{0}':\n"
    b = "        return {0}\n"
    result = ""
    for i in range(len(tags)):
        result += a.format(tags[i]) 
        result += b.format(i+1)
    with open('template/generate_tfrecord.template') as f:
        template = f.read();
    with open(os.path.join(source_root,'generate_tfrecord.py'), 'w') as x_file:
        x_file.write(template.replace("<replace with tags>", result))

def read_temp2(tags, source_root):

    pbtxt_path = os.path.join(source_root,"label_map.pbtxt")
    trainrecord_path = os.path.join(source_root,"train.record")
    testrecordpath = os.path.join(source_root,"test.record")
    with open('template/pipeline_v2.template') as f:
        template = f.read();
        template = template.replace("<tag number>", str(len(tags)) )
        template = template.replace("<pbtxt path>", pbtxt_path)
        template = template.replace("<train record path>", trainrecord_path)
        template = template.replace("<test record path>", testrecordpath)
    with open(os.path.join(source_root,'pipeline_v2.config'), 'w') as x_file:
        x_file.write(template)
        
def read_temp3(tags, source_root):
    a = "item {{\n  id: {0}\n  name: '{1}'\n}}\n"
    result=""
    for i in range(len(tags)):
        result += a.format(i+1, tags[i]) 
    with open(os.path.join(source_root,'label_map.pbtxt'), 'w') as x_file:
        x_file.write(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_root", help="image source folder root", required=True, default=DEFAULT_IMAGE_ROOT )
    args = parser.parse_args()
    tags = read_tags(args.source_root)
    read_temp1(tags, args.source_root)
    read_temp2(tags, args.source_root)
    read_temp3(tags, args.source_root)

main()
        

        

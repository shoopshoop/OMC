import sys
from OpenMonkey import *

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('usage: python3 ./display_annotation.py <path-to-annotation-file> <path-to-associated-img-directory>')
    exit()
  ann_file = sys.argv[1]
  img_dir = sys.argv[2]
  om = OpenMonkey(ann_file, img_dir)
  om.showAnns(om.imgs, om.landmarks, om.bbox, True)

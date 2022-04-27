import sys
from OpenMonkey import *

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('usage: python3 ./display_annotation.py <path-to-annotation-file> <path-to-associated-img-directory> <path-to-target-directory>')
    exit()
  ann_file = sys.argv[1]
  img_dir = sys.argv[2]
  target_dir = sys.argv[3]
  om = OpenMonkey(ann_file, img_dir)
  '''
  generate cropped images and annotations
  '''
  om.cropImgs(write=True, directory=target_dir)
  om.write_landmarks_to_file(crop=True, filename=target_dir + '/annotation_cropped.json')
  '''
  # To verify implementation
  om.write_landmarks_to_file(relative=True, filename=target_dir + '/annotation_uncropped.json')
  '''

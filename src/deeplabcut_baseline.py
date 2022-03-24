from OpenMonkey import *
import deeplabcut

'''
Following instructions found at
https://github.com/DeepLabCut/DeepLabCut/wiki/Using-labeled-data-in-DeepLabCut-that-was-annotated-elsewhere
'''

class DeepLabCut:
  def __init__(self):
    pass

  def create_new_project(self):
    deeplabcut.create_new_project('omc_deeplabcut', 'anthony',['./dummy_video.avi'], copy_videos=True, multianimal=False)

  def generate_csv(self, ann_file, img_dir):
    om = OpenMonkey(ann_file, img_dir)
    bodyparts = ['right_eye','left_eye','nose','head','neck','right_shoulder',
                 'right_elbow','right_wrist','left_shoulder','left_elbow',
                 'left_wrist','hip','right_knee','right_ankle','left knee','left_ankle','tail']
    img_paths = [img_dir + om.imgs[i] for i in range(len(om.imgs))]
    landmark_coords = [om.landmarks[i] for i in range(len(om.landmarks))]
    filename = "CollectedData_anthony.csv"
    try:
      f = open(filename, "w")
      # header info
      header = 'scorer,' + ",".join(['anthony'] * (2 * len(bodyparts))) + '\n'
      header += 'bodyparts,' +  ",".join([bp + "," + bp for bp in bodyparts]) + '\n'
      header += 'coords,' + ",".join(['x','y'] * len(bodyparts))
      f.write(header + '\n')
      for i, img_path in enumerate(img_paths):
        f.write('labeled-data/dummy_video/' + img_path[img_path.rfind('/') + 1:] + "," +\
                ",".join([str(val) for val in landmark_coords[i]]) + "\n")
      f.close()
    except:
      print("Error: could not write to file")

if __name__ == '__main__':
  config = '/home/anthony/UMN_3-3-2022/CSCI_5561/final_project/src/omc_deeplabcut-anthony-2022-03-23/config.yaml'
  dlc = DeepLabCut()
  # step 1
  # create new project
  '''
  config = dlc.create_new_project()
  '''
  '''
  be sure to edit the config.yaml file to include correct
  bodyparts and skeleton information
  '''
  # step 4
  # convert labeled data to appropriate csv format
  '''
  data_root = '../data/'
  ann_file = data_root + 'train_annotation.json'
  img_dir = data_root + 'train/'
  dlc.generate_csv(ann_file, img_dir)
  # make sure to move the csv file to labeled-data/dummy_video/
  '''
  # step 5
  '''
  # convert csv to h5 file format
  deeplabcut.convertcsv2h5(config, scorer='anthony') 
  '''
  '''
  note that all img files need to be in labeled-data/dummy_video/
  this can be done with symbolic links
  > find ./path/to/train/ -name \*.jpg -exec ln -s "{}" . ';'
  '''
  '''
  # create training dataset
  deeplabcut.create_training_dataset(config, net_type='resnet_50')
  '''
  # train model
  deeplabcut.train_network(config)

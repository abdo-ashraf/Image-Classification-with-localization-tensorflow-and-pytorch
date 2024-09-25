from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A
import tensorflow as tf

def parse_img_annot(imgpath, annotpath):
  '''
  Parameters
  ----------
  imgpath: path of image
         annotpath: Directory path of annotation matlab file

  Returns
  -------
  filename, image height, image width, class name, box coordinates
  '''
  mat = loadmat(annotpath)
  y1, y2, x1, x2 = mat['box_coord'].ravel()
  imgpath = imgpath.replace('\\', '/')
  class_name = imgpath.split('/')[-2]
  height, width = plt.imread(imgpath).shape[:2]
  filename = '/'.join(imgpath.split('/')[-2:])

  return [filename, width, height, class_name, x1, y1, x2, y2]


def dir_to_df(img_dir: str, annot_dir: str, imgfolders: list, annotfolders: list, labels_dist, is_first=True) -> pd.DataFrame:
  """ parse images and mat files of each one in folders if exists

  Parameters
  ----------
  img_dir : str
          path of images directory
  annot_dir : str
          path of annotations directory
  folders : list
          list of folders needed to parse
  labels_dist: str
          labels path in case you have the path of labels data
  is_first: bool default true
          bool flag indicates if you have your labels path or need to parse data folders
  Returns
  -------
    pandas.DataFrame
        a dataframe that contains images data
  """
  if is_first:
    demo_df = {'filename':[], 'width':[], 'height':[], 'class':[], 'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[]}

    for ifolder, afolder in zip(imgfolders, annotfolders):
      cur_img_dir = os.path.join(img_dir, ifolder)
      cur_annot_dir = os.path.join(annot_dir, afolder)
      img_files = sorted(os.listdir(cur_img_dir))
      mat_files = sorted(os.listdir(cur_annot_dir))

      for mat_file, img_file in zip(mat_files,img_files):
        mat_file_path = os.path.join(cur_annot_dir, mat_file)
        img_file_path = os.path.join(cur_img_dir, img_file)
        filename, width, height, class_name, x1, y1, x2, y2 = parse_img_annot(img_file_path, mat_file_path)

        demo_df['filename'].append(filename)
        demo_df['width'].append(width)
        demo_df['height'].append(height)
        demo_df['class'].append(class_name)
        demo_df['xmin'].append(x1)
        demo_df['ymin'].append(y1)
        demo_df['xmax'].append(x2)
        demo_df['ymax'].append(y2)

    return pd.DataFrame(demo_df)
  else:
    return pd.read_csv(labels_dist)
  

def show_one(ax, img_list: list,
              dataset_path: str = '/content/drive/MyDrive/Colab Notebooks/DataSets/CALTECH/CALTECH_Dataset/'):
    filename, w, h, class_name, x1,y1,x2,y2 = list(img_list)
    img = cv2.imread(os.path.join(dataset_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img,(x1, y1), (x2, y2),(255,0,0),2)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(class_name)
    return


def apply_agu_one(img_series : pd.Series, agu_pip : A.Compose,
                   data_path='/content/drive/MyDrive/Colab Notebooks/DataSets/CALTECH/CALTECH_Dataset'):
  
  filename = img_series['filename']
  class_labels = [img_series['class']]
  bb=[img_series['xmin'],img_series['ymin'],img_series['xmax'],img_series['ymax']]

  img_path = os.path.join(data_path, filename)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  agu_img_dict = agu_pip(image=img, bboxes=[bb], class_labels=class_labels)

  return agu_img_dict


def apply_agu_all(img_df: pd.DataFrame, agu_pip : A.Compose,save_dir : str,
                  data_path : str,
                  folder_name='Agumented'):
  
  data_path = data_path.replace('\\', '/')

  save_dir = os.path.join(save_dir, folder_name)
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  demo_df = {'filename':['demo'], 'width':[0], 'height':[0], 'class':['demo'], 'xmin':[0], 'ymin':[0], 'xmax':[0], 'ymax':[0]}
  df_agu = pd.DataFrame(demo_df)
  df_agu.drop(index=0, inplace=True)

  for c in img_df['class'].unique():
    if not os.path.exists(os.path.join(save_dir, c)):
      os.mkdir(os.path.join(save_dir, c))

  for i in range(img_df.shape[0]):
    img = img_df.iloc[i]
    agu_img_dict = apply_agu_one(img, agu_pip, data_path)

    agu_filename = img['filename'].replace('image', folder_name.lower())
    agu_img = agu_img_dict['image']
    agu_bbox = np.array(agu_img_dict['bboxes'][0], dtype='int')
    agu_class = agu_img_dict['class_labels'][0]
    agu_height, agu_width = agu_img.shape[:2]

    df_agu.loc[len(df_agu)] = [agu_filename, agu_height, agu_width, agu_class, agu_bbox[0], agu_bbox[1], agu_bbox[2], agu_bbox[3]]

    img_dir = save_dir + '/' + agu_filename
    plt.imsave(img_dir, agu_img)

  return df_agu


def load_dataset_and_bboxes(df, dataset_dir, normalize = False):
    images = []
    filenames = df['filename']
    class_name = df['class']
    l = ['xmin','ymin','xmax','ymax']
    bboxes = np.array(df[l])
    for filename in filenames:
        img = tf.keras.utils.load_img(os.path.join(dataset_dir, filename))
        img = tf.keras.utils.img_to_array(img, dtype='float64')
        if normalize:
            img = img/255
        images.append(img)

    return np.array(images), class_name, bboxes


def plot(H ,var1, var2, plot_name):
  # Get the loss metrics from the trained model
  c1 = H.history[var1]
  c2 = H.history[var2]
 
  epochs = range(len(c1))
   
  # Plot the metrics
  
  plt.plot(epochs, c1, 'b', label=var1)
  plt.plot(epochs, c2, 'r', label=var2)
  plt.title(str(plot_name))
  plt.legend()
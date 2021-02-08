import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random, choice
from models.model import Model
import os

                            
def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def get_random_subset(names, labels, percent):
    """
    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class Dataset(data.Dataset):
    def __init__(self, names, labels, args, AdaIN_model=None, img_transformer=None, tile_transformer=None, tile_ordered_transformer=None, img_final_transformer=None):
        
        self.data_path = args.path_dataset
        self.names = names
        self.labels = labels

        self.beta_parameter = args.beta_parameter
        self.grid_cell_size = args.n_tiles
        self.tile_dim = 0

        self._image_transformer = img_transformer
        self._tile_transformer = tile_transformer
        self._image_final_transformer = img_final_transformer
        self._tile_ordered_transformer = tile_ordered_transformer

        self.is_rotation = args.is_rotation
        self.is_AdaIN = args.is_AdaIN
        self.permutations = self.retrieve_permutations(args.n_tiles ** 2, args.n_jigsaw_classes)

        if args.is_AdaIN:
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self.model = AdaIN_model
          self.images = []

          if args.is_DA == True:
            self.styles = args.source + [args.target] 
          
          else:
            self.styles = args.source

    def get_one_tile(self, img, tile_number, is_ordered=False):
        
        region = ((int(tile_number%self.grid_cell_size))*(self.tile_dim),
                  (int(tile_number/self.grid_cell_size))*(self.tile_dim),
                  (int(tile_number%self.grid_cell_size)+1)*(self.tile_dim),
                  (int(tile_number/self.grid_cell_size)+1)*(self.tile_dim))

        if self.is_AdaIN == True:
          tile_style = np.random.randint(len(self.styles))
          img = self.images[tile_style]
        
        tile = img.crop(region)
        if is_ordered:
          tile = self._tile_ordered_transformer(tile)
        else:
          tile = self._tile_transformer(tile)
        return tile, region

    def __getitem__(self, index):
        
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)
        
        if self.is_rotation:
          new_img, sstask_l = self.do_rotation(img)
        
        else:
          new_img, sstask_l = self.create_jigsaw_puzzle(img, index)
        
        return new_img, sstask_l, self.labels[index]

    def do_AdaIN(self, img, index):    
        
        self.images = []
        current_style = (self.names[index].split('/'))[2]
        
        for i in range(len(self.styles)):
          
          if self.styles[i] != current_style:

            path = self.data_path+'/PACS/kfold/'+self.styles[i]
            folder = choice(os.listdir(path)) 
            img_name =  choice(os.listdir(path+'/'+folder))
            s= Image.open(path+'/'+folder+'/'+img_name)

            c_tensor = self._image_final_transformer(img).unsqueeze(0).to(self.device)
            s_tensor = self._image_final_transformer(s).unsqueeze(0).to(self.device)
            with torch.no_grad():
              out = self.model.generate(c_tensor, s_tensor, 1.0)
            out = denorm(out, self.device)
            out = out.to('cpu')
            out = np.reshape(out, (out.size()[1], out.size()[2], out.size()[3]))
            toPIL = transforms.ToPILImage()
            out = toPIL(out)
            self.images.append(out)
          
          else:
            self.images.append(img) 

    def do_rotation(self, img):
        
        if self.beta_parameter > random():
          rot_class = 0
        else:
          rot_class = np.random.randint(3) + 1
          img = img.rotate(90 * rot_class)

        img = self._image_final_transformer(img)
        return img, rot_class

    def create_jigsaw_puzzle(self, img, index):
      
      if self.is_AdaIN:

        if self.beta_parameter > random():
          jig_class = 0
        
        else:
          jig_class = np.random.randint(1, len(self.permutations))
      
        if self.is_AdaIN == True:
          self.do_AdaIN(img, index)

        self.tile_dim = int(img.size[0]/self.grid_cell_size)
        total_grid_size = self.grid_cell_size * self.grid_cell_size
        tiles = [None] * total_grid_size
        regions = [None] * total_grid_size
        
        for tile_number in range(total_grid_size):
            tiles[tile_number], regions[tile_number] = self.get_one_tile(img, tile_number, jig_class==0)
        
        new_img = Image.new('RGB', (self.tile_dim * self.grid_cell_size, self.tile_dim * self.grid_cell_size)) 
        for index in range(total_grid_size):
            new_img.paste(tiles[self.permutations[jig_class][index]], regions[index])
        
        new_img = new_img.resize(img.size)

        new_img = self._image_final_transformer(new_img)

        return new_img, int(jig_class)
      
      else:

        if self.beta_parameter > random():
          jig_class = 0
        
        else:
          jig_class = np.random.randint(len(self.permutations)) + 1
      
        if jig_class != 0:
          if self.is_AdaIN == True:
            self.do_AdaIN(img, index)

          self.tile_dim = int(img.size[0]/self.grid_cell_size)
          total_grid_size = self.grid_cell_size * self.grid_cell_size
          tiles = [None] * total_grid_size
          regions = [None] * total_grid_size
          
          for tile_number in range(total_grid_size):
              tiles[tile_number], regions[tile_number] = self.get_one_tile(img, tile_number)
          
          new_img = Image.new('RGB', (self.tile_dim * self.grid_cell_size, self.tile_dim * self.grid_cell_size)) 
          for index in range(total_grid_size):
              new_img.paste(tiles[self.permutations[jig_class - 1][index]], regions[index])
          
          new_img = new_img.crop((0,0,self.tile_dim*self.grid_cell_size,self.tile_dim*self.grid_cell_size))
          new_img = new_img.resize(img.size)
          current_style = (self.names[index].split('/'))[2]

        else:
          new_img = img

        new_img = self._image_final_transformer(new_img)

        return new_img, int(jig_class)

    def __len__(self):
        return len(self.names)

    def retrieve_permutations(self, permutations_len, n_permutations):
        
        if self.is_AdaIN:
          permutations = np.load(
              '/content/MLAI_project/utils/pytorch_permutations/permutations/permutations_hamming_9_31.npy')
        else:
          permutations = np.load(
              '/content/MLAI_project/utils/pytorch_permutations/permutations/permutations_hamming_%d_%d.npy' % (permutations_len,n_permutations))

        return permutations

class TestDataset(Dataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        # for the test dataset all the images have 0 for the jigen_class_label since they are not shuffled
        return img, 0, int(self.labels[index])

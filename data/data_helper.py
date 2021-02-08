from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.DatasetLoader import Dataset, TestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]

available_datasets = pacs_datasets

def get_train_dataloader(args, AdaIN_model):

    dataset_list = args.source
    assert isinstance(dataset_list, list)

    datasets = []
    val_datasets = []
    
    img_transformer = get_train_transformers(args)
    val_trasformer = get_val_transformer(args)
    tile_transformer = get_tile_transformer(args)
    tile_ordered_transformer = get_ordered_tile_transformer(args)
    img_final_transformer = get_image_final_transformer()

    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', dname+'.txt'), args.val_size)

        train_dataset = Dataset(name_train, labels_train, args, AdaIN_model, img_transformer=img_transformer, tile_transformer=tile_transformer, tile_ordered_transformer=tile_ordered_transformer,
                      img_final_transformer=img_final_transformer)
        datasets.append(train_dataset)

        val_dataset = TestDataset(name_val, labels_val, args, img_transformer=val_trasformer)
        val_datasets.append(val_dataset)

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)

    if args.is_AdaIN == True:
      loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    else:
      loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return loader, val_loader

def get_target_dataloader(args, AdaIN_model):

    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.target+'.txt'))
    img_transformer = get_train_transformers(args)
    tile_transformer = get_tile_transformer(args)
    tile_ordered_transformer = get_ordered_tile_transformer(args)
    img_final_transformer = get_image_final_transformer()
    

    val_dataset = Dataset(names, labels, args, AdaIN_model, img_transformer=img_transformer, tile_transformer=tile_transformer, tile_ordered_transformer=tile_ordered_transformer,
                   img_final_transformer=img_final_transformer)
    dataset = ConcatDataset([val_dataset])
    if args.is_AdaIN == True:
      loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    else:
      loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return loader

def get_val_dataloader(args):

    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.target+'.txt'))
    img_tr = get_val_transformer(args)

    val_dataset = TestDataset(names, labels, args, img_transformer=img_tr)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return loader


def get_train_transformers(args):
    
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
        
    if args.is_rotation and args.is_grayscale:
        if args.random_grayscale > 0.0:
            img_tr.append(transforms.RandomGrayscale(args.random_grayscale))
        

    return transforms.Compose(img_tr)

def get_image_final_transformer():
  
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    return transforms.Compose([transforms.ToTensor(),
                            normalize])

def get_ordered_tile_transformer(args):

    tile_tr = []  
    if args.random_grayscale > 0.0:
      tile_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    return transforms.Compose(tile_tr)

def get_tile_transformer(args):
  
    tile_tr = []  
    if args.random_grayscale > 0.0:
      tile_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    dim_to_resize = int(args.image_size / args.n_tiles)
    tile_tr.append(transforms.RandomCrop(int(((dim_to_resize)* 64)/75)))
    tile_tr.append(transforms.Resize((dim_to_resize, dim_to_resize)))

    return transforms.Compose(tile_tr)

def get_val_transformer(args):

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)
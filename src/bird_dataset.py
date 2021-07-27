import numpy as np
import pickle
import pandas as pd
import os
class BirdDataset():
    def __init__(self, data_dir='CUB_200_2011/', attr_file='attributes', species_file='classes.txt', save=False, save_file='images.pkl', preload=False, preload_dir='processed_data/', preload_file='images.pkl'):
        self.attr_file = attr_file
        self.species_file = species_file
        self.images = {}
        self.data_dir = data_dir
        self.img_dir = self.data_dir + 'images/'
        self.parts = self.get_parts()
        self.attributes = self.get_attributes()
        self.species = self._get_species_dict()
        self.bird_to_fam, self.fam = self._get_family_dict()
        
        if preload:
            self.images = pickle.load(open(preload_dir+preload_file, 'rb'))
        else:
            self._create_img_dict()
            if save:
                pickle.dump(self.images, open(preload_dir+save_file, 'wb'))
#         self.train_indices = np.random.choice(list(self.images.keys()), size=int(len(self.images.keys())*.8), replace=False)
#         self.test_indices = list(set(list(self.images.keys())) - set(list(self.train_indices)))

    def _get_species_dict(self):
        species_dict = {}
        with open(f'{self.data_dir}/{self.species_file}') as f:
            for line in f.readlines():
                line_lst = line.split()
                line_lst[1] = line_lst[1].split('.')[-1]
                species_dict[int(line_lst[0])] = line_lst[1]
        return species_dict
    
    def _get_family_dict(self):
        bird_names = pd.Series([i.split('_')[-1] for i in os.listdir(self.data_dir+'images')]).value_counts()
        bird_families = bird_names[bird_names > 1]
        bird_fam_dict = {i.split('.')[1]: i.split('_')[-1] for i in os.listdir(self.data_dir+'images')}
        fam_dict = dict(zip(bird_families.index, range(len(bird_families))))
        return bird_fam_dict, fam_dict
        
    def _create_img_dict(self):
        # Initialize dict of dicts, get filepaths
        with open(f'{self.data_dir}images.txt') as f:
            for line in f.readlines():
                line_lst = line.split()
                self.images[int(line_lst[0])] = {}
                self.images[int(line_lst[0])]['filepath'] = line_lst[1]
                self.images[int(line_lst[0])]['species_name'] = line_lst[1].split('.')[1].split('/')[0]

        # Get class labels for each img_id
#         with open(f'{self.data_dir}/image_class_labels.txt') as f:
#             for line in f.readlines():
#                 line_lst = line.split()
#                 self.images[int(line_lst[0])]['class_label'] = int(line_lst[1])-1
        rev_species_dict = {val: key for key, val in self.species.items()}
        for i in self.images.keys():
            try:self.images[i]['class_label'] = rev_species_dict[self.images[i]['species_name']]
            except KeyError:
                continue
    
        # Get bounding_box of the bird for each img_id
        with open(f'{self.data_dir}/bounding_boxes.txt') as f:
            for line in f.readlines():
                line_lst = line.split()
                self.images[int(line_lst[0])]['bounding_box'] = [float(i) for i in line_lst[1:]]

        # Get bounding_box of each part of bird for each img_id (new sub-dictionary needed)
        with open(f'{self.data_dir}/parts/part_locs.txt') as f:
            for line in f.readlines():
                line_lst = line.split()
                img_id, part_id, visible = int(line_lst[0]), int(line_lst[1]), int(line_lst[-1])
                self.images[img_id]['parts'] = self.images[img_id].get('parts', {})
                if visible == 1:
                    loc = [float(i) for i in line_lst[2:4]]
                    self.images[img_id]['parts'][self.parts[part_id]] = loc
        # get attribute labels, with the following format: <image_id> <attribute_id> <is_present> <certainty_id> <time>
        with open(self.data_dir+'attributes/image_attribute_labels.txt') as f:
            # for now, i'm not considering certainty values when inserting attributes into the dictionary
            for line in f.readlines():
                line_lst = line.split()
                img_id, attr_id, present = int(line_lst[0]), int(line_lst[1]), int(line_lst[2])
                self.images[img_id]['attributes'] = self.images[img_id].get('attributes', [])
                if present == 1:
                    self.images[img_id]['attributes'].append(self.attributes[attr_id])
        for i in self.images.keys():
            self.images[i]['image_id'] = i
            
    def get_parts(self):
        parts = {}
        with open(self.data_dir+'parts/parts.txt') as f:
            for line in f.readlines():
                line_lst = line.split()
                parts[int(line_lst[0])] = ' '.join(line_lst[1:])
        return parts
    
    def get_attributes(self):
        attributes = {}
        with open(self.data_dir+f'attributes/{self.attr_file}.txt') as f:
            for line in f.readlines():
                line_lst = line.split()
                attributes[int(line_lst[0])] = line_lst[1]
        return attributes
    
    def open_image(self, img_id):
        return Image.open(self.img_dir+self.images[img_id]['filepath'])
    
    def draw_bbox(self, img_id):
        img = self.open_image(img_id)
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        bbox = self.images[img_id]['bounding_box']
        rect = patches.Rectangle(tuple(bbox[:2]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        
        ax.add_patch(rect)
        plt.show()
    
    def plot_parts_bbox(self, img_id):
        img = self.open_image(img_id)
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        
        xy = self.images[img_id]['parts'].values()
        
        x = [i[0] for i in xy]
        y = [i[1] for i in xy]
        ax.scatter(x, y)
        
        parts = self.images[img_id]['parts'].keys()
        for i, txt in enumerate(parts):
            ax.annotate(txt, (x[i], y[i]))
            
        bbox = self.images[img_id]['bounding_box']
        rect = patches.Rectangle(tuple(bbox[:2]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        
        ax.add_patch(rect)
        plt.show()
    
    def get_attributes_img(self, img_id):
        return self.images[img_id]['attributes']
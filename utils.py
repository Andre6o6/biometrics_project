import numpy as np
import os, os.path
from skimage import io

def classify(clf, img, classes):
    '''
    Find image among classes that is closest to img.
    '''
    distances = np.array(list(map(lambda img2: clf.Distance(img, img2), classes)))
    distances = distances / np.sum(distances)
    
    return distances.argmin()

def classify_many(clf, imgs, classes, vote='soft'):
    
    all_dist = []
    for img in imgs:
        distances = np.array(list(map(lambda img2: clf.Distance(img, img2), classes)))
        distances = distances / np.sum(distances)
        all_dist.append(distances)
        
    if vote=='soft':
        return np.mean(all_dist, axis=0).argmin()
    else:
        votes = np.argmin(all_dist, axis=1)
        return np.bincount(votes).argmax()

def rank(clf, img, label, classes):
    distances = np.array(list(map(lambda img2: clf.Distance(img, img2), classes)))
    distances = distances / np.sum(distances)
    
    dist = distances[label]
    r = sorted(distances).index(dist)
    
    return r

def load_from_folders(path, from_idx=0, to_idx=11):
    '''
    Load and split into train (images to be recognised) and test (class samples).
    '''

    train = []
    labels = []
    test = []

    i = 0
    for dir in os.listdir(path)[from_idx:to_idx]:
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            subj_images = []
            
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                img = io.imread(file_path)
                subj_images.append(img)
            
            #random.shuffle(subj_images)
            
            test.append(subj_images[0])    # Save 1 image as an example of class
            train.extend(subj_images[1:])  # Add other images to train set ...
            labels.extend([i for x in subj_images[1:]])    # ... and save their class number
            
            i +=1
            
    train = np.array(train)
    test = np.array(test)
    
    return train, labels, test
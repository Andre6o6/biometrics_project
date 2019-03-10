import numpy as np
import random
from skimage.transform import rescale
from scipy.fftpack import dct, idct
from scipy.ndimage import sobel

def cosine_similarity(a,b):
    '''
    Косинусовая мера близости. Для тестов.
    '''
    return np.dot(a.reshape(-1), b.reshape(-1))/(np.linalg.norm(a)*np.linalg.norm(b))

def distance(a,b,mode='euc'):
    if mode=='euc':
        return np.linalg.norm(a - b)
    elif mode=='abs':
        return np.mean(np.abs(a.reshape(-1) - b.reshape(-1)))

class ScaleClassifier:
    '''
    Классификатор, сравнивающий уменьшенные изображения.
    Усредняет расстояния, полученные для разных scale-ов.
    '''
    def __init__(self, scales=[2,3,4]):
        self.scales = scales

    def Distance(self, img1, img2):
        dist = 0
        
        for s in self.scales:
            img1_rescaled = rescale(img1, 1/s, mode='constant', anti_aliasing=False, multichannel=False)
            img2_rescaled = rescale(img2, 1/s, mode='constant', anti_aliasing=False, multichannel=False)
            
            dist += np.linalg.norm(img1_rescaled - img2_rescaled)
        
        dist /= len(self.scales)
        return dist


class RandomPointsClassifier:
    '''
    Классификатор, который выбирает n_points рандомных точек и смотрит, 
    на сколько изображения отличаются в этих точках. 
    '''
    def __init__(self, n_points=100, img_size=(112, 92), random_state=0):
        random.seed(random_state)
        
        h,w = img_size
        
        self.points = np.zeros((n_points,2), dtype=int)
        for i in range(n_points):
            self.points[i] = (random.randint(0, h-1), random.randint(0, w-1))
    
    def Distance(self, img1, img2):
        points1 = img1[self.points[:, 0], self.points[:, 1]]
        points2 = img2[self.points[:, 0], self.points[:, 1]]
        return np.linalg.norm(points1 - points2)


class DCTClassifier:
    '''
    Классификатор, сравнивающий результаты применения DCT к изображениям.
    Использует не всю полученную матрицу, а только левый верхний квадрат размера size.
    '''
    def __init__(self, size=10):
        self.size = size
        
    def Distance(self, img1, img2):
        img1_dct = dct(dct(img1.T, norm='ortho').T, norm='ortho')
        img1_dct = img1_dct[:self.size, :self.size]
        img2_dct = dct(dct(img2.T, norm='ortho').T, norm='ortho')
        img2_dct = img2_dct[:self.size, :self.size]
        return np.linalg.norm(img1_dct - img2_dct)
    

class DFTClassifier:
    '''
    Классификатор, сравнивающий результаты применения DFT к изображениям.
    Если clipped=True, используется только левыя верхняя четверть спектрограммы.
    '''
    def __init__(self, clipped=True, size=None):
        self.clipped = clipped
        self.size = size
        
    def Distance(self, img1, img2):
        img1_dft = np.fft.fft2(img1)
        img1_dft = np.fft.fftshift(img1_dft)
        img1_dft = 20*np.log(np.abs(img1_dft))
        
        img2_dft = np.fft.fft2(img2)
        img2_dft = np.fft.fftshift(img2_dft)
        img2_dft = 20*np.log(np.abs(img2_dft))
        
        if self.clipped:
            h,w = img2_dft.shape
            img1_dft = img1_dft[:h//2, :w//2]
            img2_dft = img2_dft[:h//2, :w//2]
            
            if self.size is not None:
                    img1_dft = img1_dft[-self.size:, -self.size:]
                    img2_dft = img2_dft[-self.size:, -self.size:]
        
        return np.linalg.norm(img1_dft - img2_dft)


class HistogramClassifier:
    '''
    Классификатор, сравнивающий гистограммы изображений.
    '''
    def __init__(self, size=10):
        self.size = size
        
    def Distance(self, img1, img2):
        img1_hist, bin_edges = np.histogram(img1, bins=self.size)
        img2_hist, bin_edges = np.histogram(img2, bins=self.size)
        return np.linalg.norm(img1_hist - img2_hist)


class GradientClassifier:
    '''
    Классификатор, сравнивающий графики градиента яркости в вертикальном направлении.
    Я хз, правильно ли я это считаю, но нам особо и не объясняли.
    '''
    def Distance(self, img1, img2):
        img1_grad = sobel(img1/255, axis=0).mean(axis=0)
        img2_grad = sobel(img2/255, axis=0).mean(axis=0)
        return np.linalg.norm(img1_grad - img2_grad, ord=1)   # L1-норма работает лучше
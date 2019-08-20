import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import ipywidgets 
from ipywidgets import interact, Dropdown, IntSlider
#%matplotlib notebook
plt.style.use('grayscale')

data_path = 'D:/Downloads/MRNet-v1.0/'
train_path = str(data_path) + 'train/'

test = np.load(train_path + 'axial/0000.npy')

train_abnl = pd.read_csv(data_path + 'train-abnormal.csv', header=None,
                       names=['Case', 'Abnormal'], 
                       dtype={'Case': str, 'Abnormal': np.int64})

# data loading functions
def load_one_stack(case, data_path=train_path, plane='coronal'):
    string = data_path + plane + '/{}.npy'
    fpath = string.format(case)
    return np.load(fpath)

def load_stacks(case, data_path=train_path):
    x = {}
    planes = ['coronal', 'sagittal', 'axial']
    for i, plane in enumerate(planes):
        x[plane] = load_one_stack(case, plane=plane)
    return x

# interactive viewer
class KneePlot():
    def __init__(self, x, figsize=(10, 10)):
        self.x = x
        self.planes = list(x.keys())
        self.slice_nums = {plane: self.x[plane].shape[0] for plane in self.planes}
        self.figsize = figsize
    
    def _plot_slices(self, plane, im_slice): 
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.imshow(self.x[plane][im_slice, :, :])
        plt.show()
    
    def draw(self):
        planes_widget = Dropdown(options=self.planes)
        plane_init = self.planes[0]
        slice_init = self.slice_nums[plane_init] - 1
        slices_widget = IntSlider(min=0, max=slice_init, value=slice_init//2)
        def update_slices_widget(*args):
            slices_widget.max = self.slice_nums[planes_widget.value] - 1
            slices_widget.value = slices_widget.max // 2
        planes_widget.observe(update_slices_widget, 'value')
        for plane in self.planes:
            for im_slice in self.slice_nums:
                interact(self._plot_slices(plane=planes_widget, im_slice=slices_widget))

    def resize(self, figsize): self.figsize = figsize

# example usage
case = train_abnl.Case[1]
x = load_stacks(case)
plot = KneePlot(x, figsize=(8, 8))
plot.draw()
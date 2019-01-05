import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

csv_path = "/Volumes/ISIC_IMAGES/GroundTruth.csv.xls"
root_path = "/Volumes/ISIC_IMAGES/images"

class LoadData(Dataset):
	'''
	Oreders images to classes
	'''
	def __init__(self, csv_path, img_path):
		'''
		Args:
			csv_file (string): path to csv label file
			img_path (string): path to images 
		'''
		self.csv_file = pd.read_csv(csv_path, header=None)
		self.img_path = img_path

	def __len__(self):
		return len(self.csv_file)

	def __getitem__(self, idx):
		# join root dir and image name which is gotten form labels file
		img_name = os.path.join(self.img_path, 
								self.csv_file.iloc[idx, 0]) + ".jpg"
		# read the image
		image = io.imread(img_name)
		diagnosis = self.csv_file.iloc[idx, 1]
		sample ={"image": image, "diagnosis": diagnosis}
		return sample

	def getClasses(self):
		'''
		returns list of all classes
		'''
		return self.csv_file.iloc[:, 1].unique()

loadImages = LoadData(csv_path=csv_path, img_path=root_path)
classes = loadImages.getClasses()

# Create folders for class names
for instance in classes:
	try:
		os.makedirs(os.path.join(root_path, instance))
	except OSError:
		if not os.path.isdir(os.path.join(root_path, instance)):
			raise

# Prints how manny images in total
print("{} total images.".format(len(loadImages)))

def save_imgs(img_data, path):
	'''
	Saves images to disk

	Args:
		img_data (dictionary): contains image file and its classification
		paht (string): path to where folders with class names are located
	'''
	print("Saving Images to disk...")
	for i,image in enumerate(img_data):
		sample, diagnosis = img_data[i]["image"], img_data[i]["diagnosis"]
		img = Image.fromarray(sample, "RGB")
		img.save(os.path.join(path, diagnosis) + "/isic_{}.jpg".format(i))
		# prints every qartile
		if i % (len(img_data) / 4) == 0:
			print("... {}%".format(100 * i / len(img_data) ))
	print("100% done!")

def show_class_structure(class_names, path):
	'''
	Prints how manny images in each of the classes

	Args:
		class_names (list): list of all classes
		path (string): path to where folders with class names are located
	'''
	for name in class_names:
		lenght = len(os.listdir(os.path.join(root_path, name)))
		print("{} images in {} class.".format(lenght, name) )

save_imgs(img_data=loadImages, path=root_path)
show_class_structure(class_names=classes, path=root_path)
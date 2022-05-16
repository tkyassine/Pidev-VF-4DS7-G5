import cv2
from PIL import Image
import numpy as np
import os
import re
import pytesseract

base_path = "../cheque/newOnes/"
DATA_AXIS = '../cheque/AXIS/'
DATA_ICICI = '../cheque/ICICI/'
DATA_SYNDICATE = '../cheque/SYNDICATE/'
DATA_CANARA = '../cheque/CANARA/'
MontantEnLettre = '../cheque/MontantEnLettre/'
MontantEnChiffre = '../cheque/MontantEnChiffre/'

def ConvertingToPNG():
  cheque = "../cheque/original/"
  if not os.path.exists(base_path):
    os.mkdir(base_path)
    print("Directory ", base_path ,  " Created ")
  else:
    print("Directory " , base_path ,  " already exists")

  for infile in os.listdir(cheque):
    if not "ipynb_checkpoints" in infile:
      print ("Converting  : " + infile + "...")
      read = cv2.imread(cheque + infile)
      outfile = infile.split('.')[0] + '.png'
      cv2.imwrite(base_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])

def center_crop(im, crop_pixels=45):
  return im[crop_pixels:im.shape[0] - crop_pixels * 3, crop_pixels:im.shape[1] - crop_pixels]

def bright_contrast_loop(image, alpha=1, beta=0):
  new_image = np.zeros(image.shape, image.dtype)
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      for c in range(image.shape[2]):
        new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
  return new_image

def imagePreprocessing():
  for infile in os.listdir(base_path):
    if not "ipynb_checkpoints" in infile:
      print("Traitement : " + infile + "...")

      image = cv2.imread(base_path+ infile)

      resized = cv2.resize(image, (2300, 1000), )

      cropped = center_crop(resized)

      contrast_im = bright_contrast_loop(cropped, alpha=1.07)

      graychiffre = cv2.cvtColor(contrast_im, cv2.COLOR_BGR2GRAY)
      thresh = cv2.threshold(graychiffre, 150, 255, cv2.THRESH_BINARY_INV)[1]
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
      result = 255 - opening

      cv2.imwrite(base_path + infile, result)

def imagePreprocessing2():
  for infile in os.listdir(base_path):
    if not "ipynb_checkpoints" in infile:
      print("Denoising  : " + infile + "...")
      image = cv2.imread(base_path + infile)

      dst = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 21, 45)

      kernel = np.ones((2, 2), np.uint8)
      erosion = cv2.erode(dst, kernel, iterations=1)

      cv2.imwrite(base_path + infile, erosion)

def repartirChequeSelonBanque():
  if not os.path.exists(DATA_AXIS):
    os.mkdir(DATA_AXIS)
    print("Directory ", DATA_AXIS, " Created ")
  else:
    print("Directory ", DATA_AXIS, " already exists")

  if not os.path.exists(DATA_ICICI):
    os.mkdir(DATA_ICICI)
    print("Directory ", DATA_ICICI, " Created ")
  else:
    print("Directory ", DATA_ICICI, " already exists")

  if not os.path.exists(DATA_SYNDICATE):
    os.mkdir(DATA_SYNDICATE)
    print("Directory ", DATA_SYNDICATE, " Created ")
  else:
    print("Directory ", DATA_SYNDICATE, " already exists")

  if not os.path.exists(DATA_CANARA):
    os.mkdir(DATA_CANARA)
    print("Directory ", DATA_CANARA, " Created ")
  else:
    print("Directory ", DATA_CANARA, " already exists")

  for infile in os.listdir(base_path):
    if not "ipynb_checkpoints" in infile:
      extract = pytesseract.image_to_string(Image.open(base_path + infile))
      image = cv2.imread(base_path+infile)
      extract = ''.join(extract)
      extract = extract.split(' ')
      l = list()
      for i in extract:
        if i != '' and i != ' ':
          l.append(i)
      print(l)
      for i in l:
        if re.search("^axi.*", i.lower()):
          cv2.imwrite(DATA_AXIS + infile, image)
        elif re.search("^synd.*", i.lower()):
          cv2.imwrite(DATA_SYNDICATE + infile, image)
        elif re.search("^icic.*", i.lower()):
          cv2.imwrite(DATA_ICICI + infile, image)
        elif re.search("^cana.*", i.lower()):
          cv2.imwrite(DATA_CANARA + infile, image)

def ExtraireMontantEnLettre():
  if not os.path.exists(MontantEnLettre):
    os.mkdir(MontantEnLettre)
    print("Directory ", MontantEnLettre, " Created ")
  else:
    print("Directory ", MontantEnLettre, " already exists")

  im = Image.open(r'../data/blank.png')
  Image2copy = im.copy()
  for infile_axis in os.listdir(DATA_AXIS):
    if not "ipynb_checkpoints" in infile_axis:
      y_axis = 245
      x_axis = 200
      h_axis = 86
      w_axis = 2000
      im_axis = cv2.imread(DATA_AXIS + infile_axis)
      crop_axis = im_axis[y_axis:y_axis + h_axis, x_axis:x_axis + w_axis]
      cv2.imwrite(MontantEnLettre + infile_axis, crop_axis)

      im_axis = Image.open(r'../data/blank.png')
      blank_axis = im_axis.copy()

      crop_axis_2 = Image.open(r'../cheque/MontantEnLettre/'+infile_axis)
      blank_axis.paste(crop_axis_2, (90, 300))
      blank_axis.save(r'../cheque/MontantEnLettre/'+infile_axis)

  for infile_canara in os.listdir(DATA_CANARA):
    if not "ipynb_checkpoints" in infile_canara:
      y_canara = 238 #255
      x_canara = 300
      h_canara = 84
      w_canara = 2000
      im_canara = cv2.imread(DATA_CANARA + infile_canara)
      crop_canara = im_canara[y_canara:y_canara + h_canara, x_canara:x_canara + w_canara]
      cv2.imwrite(MontantEnLettre + infile_canara, crop_canara)

      im_canara = Image.open(r'../data/blank.png')
      blank_canara = im_canara.copy()

      crop_canara_2 = Image.open(r'../cheque/MontantEnLettre/' + infile_canara)
      blank_canara.paste(crop_canara_2, (90, 300))
      blank_canara.save(r'../cheque/MontantEnLettre/' + infile_canara)

  for infile_icici in os.listdir(DATA_ICICI):
    if not "ipynb_checkpoints" in infile_icici:
      y_icici = 240
      x_icici = 300
      h_icici = 83
      w_icici = 2000
      im_icici = cv2.imread(DATA_ICICI + infile_icici)
      crop_icici = im_icici[y_icici:y_icici + h_icici, x_icici:x_icici + w_icici]
      cv2.imwrite(MontantEnLettre + infile_icici, crop_icici)

      im_icici = Image.open(r'../data/blank.png')
      blank_icici = im_icici.copy()

      crop_icici_2 = Image.open(r'../cheque/MontantEnLettre/' + infile_icici)
      blank_icici.paste(crop_icici_2, (90, 300))
      blank_icici.save(r'../cheque/MontantEnLettre/' + infile_icici)

  for infile_syndicate in os.listdir(DATA_SYNDICATE):
    if not "ipynb_checkpoints" in infile_syndicate:
      y_syndicate = 258
      x_syndicate = 300
      h_syndicate = 83
      w_syndicate = 2000
      im_syndicate = cv2.imread(DATA_SYNDICATE + infile_syndicate)
      crop_syndicate = im_syndicate[y_syndicate:y_syndicate + h_syndicate, x_syndicate:x_syndicate + w_syndicate]
      cv2.imwrite(MontantEnLettre + infile_syndicate, crop_syndicate)

      im_syndicate = Image.open(r'../data/blank.png')
      blank_syndicate = im_syndicate.copy()

      crop_syndicate_2 = Image.open(r'../cheque/MontantEnLettre/' + infile_syndicate)
      blank_syndicate.paste(crop_syndicate_2, (90, 300))
      blank_syndicate.save(r'../cheque/MontantEnLettre/' + infile_syndicate)


def ExtraireMontantEnChiffre():
  if not os.path.exists(MontantEnChiffre):
    os.mkdir(MontantEnChiffre)
    print("Directory ", MontantEnChiffre, " Created ")
  else:
    print("Directory ", MontantEnChiffre, " already exists")

  for infile_axis in os.listdir(DATA_AXIS):
    if not "ipynb_checkpoints" in infile_axis:
      y_axis = 350
      x_axis = 1670
      h_axis = 74
      w_axis = 370
      im_axis = cv2.imread(DATA_AXIS + infile_axis)
      crop_axis = im_axis[y_axis:y_axis + h_axis, x_axis:x_axis + w_axis]
      cv2.imwrite(MontantEnChiffre + infile_axis, crop_axis)

  for infile_canara in os.listdir(DATA_CANARA):
    if not "ipynb_checkpoints" in infile_canara:
      y_canara = 340
      x_canara = 1780
      h_canara = 90
      w_canara = 310
      im_canara = cv2.imread(DATA_CANARA + infile_canara)
      crop_canara = im_canara[y_canara:y_canara + h_canara, x_canara:x_canara + w_canara]
      cv2.imwrite(MontantEnChiffre + infile_canara, crop_canara)

  for infile_icici in os.listdir(DATA_ICICI):
    if not "ipynb_checkpoints" in infile_icici:
      y_icici = 348
      x_icici = 1751
      h_icici = 74
      w_icici = 300
      im_icici = cv2.imread(DATA_ICICI + infile_icici)
      crop_icici = im_icici[y_icici:y_icici + h_icici, x_icici:x_icici + w_icici]
      cv2.imwrite(MontantEnChiffre + infile_icici, crop_icici)

  for infile_syndicate in os.listdir(DATA_SYNDICATE):
    if not "ipynb_checkpoints" in infile_syndicate:
      y_syndicate = 330
      x_syndicate = 1725
      h_syndicate = 95
      w_syndicate = 345
      im_syndicate = cv2.imread(DATA_SYNDICATE + infile_syndicate)
      crop_syndicate = im_syndicate[y_syndicate:y_syndicate + h_syndicate, x_syndicate:x_syndicate + w_syndicate]
      cv2.imwrite(MontantEnChiffre + infile_syndicate, crop_syndicate)
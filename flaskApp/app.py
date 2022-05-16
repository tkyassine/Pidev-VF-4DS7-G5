from flask import Flask,request, url_for, redirect, render_template,flash
import urllib.request
import argparse
import os
from typing import Tuple, List
from werkzeug.utils import secure_filename
from pprint import pprint
import cv2
from PIL import Image
import numpy as np
import re
import pytesseract
from segmentation import detectionPage, detectionWord, sort_words
from model import Model, DecoderType
from preprocessor import Preprocessor
from dataloader_iam import Batch
import difflib
from spellchecker import SpellChecker
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from matplotlib.pyplot import plot
import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tensorflow import keras
import tensorflow as tf

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif', 'tif']

UPLOAD_FOLDER = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/uploads'
AXIS_BANK = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/axis/'
CANARA_BANK = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/canara/'
ICICI_BANK = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/icici/'
SYNDICATE_BANK = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/syndicate/'
LETTRE = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/'
BLANK_IMAGE = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/blank.png'

app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
Montant_chiffre=0
lists=[]
class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_number_list = '../model/NumberList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def preprocessing_before_crop(image_path):
    img_path = image_path
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    (b, g, r) = image[100, 1000]
    for i in range(0, 186):
        for j in range(0, 1792):
            image[i, j] = (b, g, r)
    (b2, g2, r2) = image[220, 2300]
    for i in range(200, 350):
        for j in range(1750, 2300):
            image[i, j] = (b2, g2, r2)
    (b3, g3, r3) = image[500, 750]
    for i in range(465, 510):
        for j in range(0, 1680):
            image[i, j] = (b3, g3, r3)
    scale_percent = 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_colred = cv2.cvtColor(cv2.resize(image, dim), cv2.COLOR_BGR2GRAY)
    image_filtred1 = cv2.bilateralFilter(image_colred, 5, 75, 75)
    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 221,
                                           12)
    image_blured = cv2.medianBlur(image_threshed,
                                  3)  # 3 - 5 is the best value so far because the 7 make the 'c' of lalch (2nd word not clear)
    cropped_image = image_blured[40:530, 200:2285]
    image_bordered = cv2.copyMakeBorder(cropped_image, 3, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image_bordered


def get_contour_precedence_amount(contour_amount, cols_amount):
    tolerance_factor_amount = 100
    origin_amount = cv2.boundingRect(contour_amount)
    return ((origin_amount[1] // tolerance_factor_amount) * tolerance_factor_amount) * cols_amount + origin_amount[0]


def amount_cropping(cropped_image):
    amount_image = cropped_image[300:490, 1450:2085]
    v_amount_image = np.median(amount_image)
    sigma_amount_image = 0.33
    lower_amount_image = int(max(0, (1.0 - sigma_amount_image) * v_amount_image))
    upper_amount_image = int(min(255, (1.0 + sigma_amount_image) * v_amount_image))
    canned_amount_image = cv2.Canny(amount_image, upper_amount_image, lower_amount_image)
    lines_amount_image = cv2.HoughLinesP(canned_amount_image, 1, np.pi / 180, 300, minLineLength=300, maxLineGap=600)
    for line in lines_amount_image:
        x1, y1, x2, y2 = line[0]
        cv2.line(amount_image, (x1, y1), (x2, y2), (255, 0, 0), 9)
    amount_image_copy = amount_image.copy()
    kernel = np.ones((1, 1), np.uint8)
    amount_image_copy_erosion = cv2.erode(amount_image_copy, kernel, iterations=3)
    contours_amount, hierarchy_amount = cv2.findContours(image=amount_image_copy_erosion, mode=cv2.RETR_TREE,
                                                         method=cv2.CHAIN_APPROX_SIMPLE)
    contours_amount.sort(key=lambda x: get_contour_precedence_amount(x, amount_image_copy_erosion.shape[1]))
    ROI_number_amount = 0
    for c in contours_amount:
        x, y, w, h = cv2.boundingRect(c)
        if 30 < h < 85 and 2 < w < 85:
            cv2.rectangle(amount_image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            ROI_amount = amount_image[y:y + h, x:x + w]
            cv2.imwrite(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/segments_chiffre/segment_{}.png'.format(ROI_number_amount), ROI_amount)
            ROI_number_amount += 1

    return amount_image

def deleteSegmentChiffre():
    base_path = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/segments_chiffre/'
    for infile in os.listdir(base_path):
        if infile.endswith('png') and infile.startswith('segment'):
            os.remove(base_path+infile)

def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

def recognitionMontantEnChiffre():
    segment_chiffre='C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/segments_chiffre/'
    model = tf.keras.models.load_model('C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/public/mod.h5')
    directory=os.listdir(segment_chiffre)
    directory.sort()
    montant_chiffre=[]
    dict_montant={}
    for segment in directory:
        print(segment)
        if not "ipynb_checkpoints" in segment:
            path = segment_chiffre + segment
            print(path)
            if segment !='segment_0.png':
                seg=load_image(path)
                seg = np.expand_dims(seg, axis=0)
                pred=model.predict(seg)
                pred_idx = np.argmax(pred)
                montant_chiffre.append(pred_idx)
                dict_montant[pred_idx]='1'
    return montant_chiffre,dict_montant

def center_crop(im, crop_pixels=45):
  return im[crop_pixels:im.shape[0] - crop_pixels * 3, crop_pixels:im.shape[1] - crop_pixels]

def bright_contrast_loop(image, alpha=1, beta=0):
  new_image = np.zeros(image.shape, image.dtype)
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      for c in range(image.shape[2]):
        new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
  return new_image

def readingMontantChiffre(montant_chif):
  for i in range(len(montant_chif)):
    montant_chif[i]=str(montant_chif[i])
  m=''.join(montant_chif)
  lists.append(m)
  Montant_chiffre = m

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def acceuil():
    return render_template('index.html')

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route("/about", methods=['GET', 'POST'])
def about():
    return render_template('aboutus.html')

@app.route("/demo", methods=['GET', 'POST'])
def demo():
    return render_template('recom.html')

@app.route("/predict", methods=['POST'])
def prediction():
    file = request.files['file']
    pprint(dir(file))

    if file.filename == '':
        flash('no image selected')
        return redirect('/demo')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect('/converting')
    else:
        flash('Allowed types are jpg jpeg png gif only !')
        return redirect('/converting')

@app.route("/converting", methods=['GET'])
def converting():
    for infile in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in infile:
            read = cv2.imread(UPLOAD_FOLDER + '/' + infile)
            outfile = infile.split('.')[0] + '.png'
            cv2.imwrite(UPLOAD_FOLDER + '/' + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
            os.remove(UPLOAD_FOLDER + '/'+infile)
            return redirect('/predictChiffre')

@app.route("/predictChiffre", methods=['GET','POST'])
def resultatChiffre():
    for infile in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in infile:
            path_to_image = UPLOAD_FOLDER + '/' + infile
            cropped_image = preprocessing_before_crop(path_to_image)
            amount_image = amount_cropping(cropped_image)
            a, b = recognitionMontantEnChiffre()
            readingMontantChiffre(a)
            deleteSegment()
            return redirect('/preprocessing')

@app.route("/preprocessing", methods=['GET'])
def preprocessing():
    for outfile in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in outfile:
            image = cv2.imread(UPLOAD_FOLDER + '/' + outfile)
            resized = cv2.resize(image, (2300, 1000), )

            cropped = center_crop(resized)

            contrast_im = bright_contrast_loop(cropped, alpha=1.07)

            graychiffre = cv2.cvtColor(contrast_im, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(graychiffre, 150, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            result = 255 - opening

            #dst = cv2.fastNlMeansDenoisingColored(result, None, 30, 30, 21, 45)

            kernel = np.ones((2, 2), np.uint8)
            erosion = cv2.erode(result, kernel, iterations=1)

            cv2.imwrite(UPLOAD_FOLDER + '/' + outfile, erosion)
            return redirect('/repartir')

@app.route("/repartir", methods=['GET'])
def repartition():
    for file in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in file:
            extract = pytesseract.image_to_string(Image.open(UPLOAD_FOLDER + '/' + file))
            image = cv2.imread(UPLOAD_FOLDER + '/' + file)
            extract = ''.join(extract)
            extract = extract.split(' ')
            l = list()
            for i in extract:
                if i != '' and i != ' ':
                    l.append(i)
            print(l)
            for i in l:
                if re.search("^axi.*", i.lower()):
                    cv2.imwrite(AXIS_BANK + file, image)
                elif re.search("^synd.*", i.lower()) or re.search("^dicat.*", i.lower()):
                    cv2.imwrite(SYNDICATE_BANK + file, image)
                elif re.search("^icic.*", i.lower()):
                    cv2.imwrite(ICICI_BANK + file, image)
                elif re.search("^cana.*", i.lower()):
                    cv2.imwrite(CANARA_BANK + file, image)
            return redirect('/extract')

@app.route("/extract", methods=['GET'])
def extraction():
    for infile_axis in os.listdir(AXIS_BANK):
        if not "ipynb_checkpoints" in infile_axis:
            y_axis = 245
            x_axis = 200
            h_axis = 86
            w_axis = 2000
            im_axis = cv2.imread(AXIS_BANK + infile_axis)
            crop_axis = im_axis[y_axis:y_axis + h_axis, x_axis:x_axis + w_axis]
            cv2.imwrite(LETTRE + infile_axis, crop_axis)

            im_axis = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/blank.png')
            blank_axis = im_axis.copy()

            crop_axis_2 = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_axis)
            blank_axis.paste(crop_axis_2, (90, 300))
            blank_axis.save(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_axis)

    for infile_canara in os.listdir(CANARA_BANK):
        if not "ipynb_checkpoints" in infile_canara:
            y_canara = 238  # 255
            x_canara = 300
            h_canara = 84
            w_canara = 2000
            im_canara = cv2.imread(CANARA_BANK + infile_canara)
            crop_canara = im_canara[y_canara:y_canara + h_canara, x_canara:x_canara + w_canara]
            cv2.imwrite(LETTRE + infile_canara, crop_canara)

            im_canara = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/blank.png')
            blank_canara = im_canara.copy()

            crop_canara_2 = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_canara)
            blank_canara.paste(crop_canara_2, (90, 300))
            blank_canara.save(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_canara)

    for infile_icici in os.listdir(ICICI_BANK):
        if not "ipynb_checkpoints" in infile_icici:
            y_icici = 240
            x_icici = 300
            h_icici = 83
            w_icici = 2000
            im_icici = cv2.imread(ICICI_BANK + infile_icici)
            crop_icici = im_icici[y_icici:y_icici + h_icici, x_icici:x_icici + w_icici]
            cv2.imwrite(LETTRE + infile_icici, crop_icici)

            im_icici = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/blank.png')
            blank_icici = im_icici.copy()

            crop_icici_2 = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_icici)
            blank_icici.paste(crop_icici_2, (90, 300))
            blank_icici.save(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_icici)

    for infile_syndicate in os.listdir(SYNDICATE_BANK):
        if not "ipynb_checkpoints" in infile_syndicate:
            y_syndicate = 258
            x_syndicate = 300
            h_syndicate = 83
            w_syndicate = 2000
            im_syndicate = cv2.imread(SYNDICATE_BANK + infile_syndicate)
            crop_syndicate = im_syndicate[y_syndicate:y_syndicate + h_syndicate, x_syndicate:x_syndicate + w_syndicate]
            cv2.imwrite(LETTRE + infile_syndicate, crop_syndicate)

            im_syndicate = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/blank.png')
            blank_syndicate = im_syndicate.copy()

            crop_syndicate_2 = Image.open(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_syndicate)
            blank_syndicate.paste(crop_syndicate_2, (90, 300))
            blank_syndicate.save(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + infile_syndicate)
    return redirect('/recognition')

def segmentationImage(img_name: str) -> None:
    image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    # Crop image and get bounding boxes r'C:\Users\yassi\Desktop\segment.png'
    assert image is not None
    crop = detectionPage(image)
    boxes = detectionWord(crop)
    lines = sort_words(boxes)

    # Saving the bounded words from the page image in sorted way
    i = 0
    for line in lines:
        text = crop.copy()
        for (x1, y1, x2, y2) in line:
            # roi = text[y1:y2, x1:x2]
            save = Image.fromarray(text[y1:y2, x1:x2])
            # print(i)
            save.save(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/segments/segment' + str(i) + '.png')
            i += 1

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def deleteSegment():
    base_path = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/segments/'
    for infile in os.listdir(base_path):
        if infile.endswith('png') and infile.startswith('segment'):
            os.remove(base_path+infile)

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def SpellingMistakeCorrection(initial_data:dict):
    all_data = {}
    list_of_words = ["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety","hundred","thousand","million","lakh","lakhs"]
    list_of_special_car = ['*', ',', '.', '/', '#', '+', '-', ';', ':', '?', '!', '%', '^', '~', '{', '}', '[', ']', '(',')', '_', '@']
    spell = SpellChecker(language=None)
    spell.word_frequency.load_text_file('../data/my_text_file.txt')
    data = []
    probs = []
    for i, j in initial_data.items():
        if '0' not in i and '1' not in i and '2' not in i and '3' not in i and '4' not in i and '5' not in i and '6' not in i and '7' not in i and '8' not in i and '9' not in i:
            all_data[i] = j
    for i in range(len(list_of_special_car)):
        if list_of_special_car[i] in all_data:
            del all_data[list_of_special_car[i]]
    for i, j in all_data.items():
        data.append(i)
        probs.append(j)

    data = [x.lower() for x in data]

    for a in range(len(data)):
        data[a]=spell.correction(data[a])

    for i in range(len(data)):
        list_ratios=[]
        if data[i] not in list_of_words:
            string_incorr = data[i]
            for j in list_of_words:
                string = j
                emp = difflib.SequenceMatcher(None,string_incorr,string)
                list_ratios.append(emp.ratio())
                max_value = max(list_ratios)
                index = list_ratios.index(max_value)
                data[i]=list_of_words[index]
    final_dict= dict(zip(data, probs))
    return final_dict

def infer(model: Model, fn_img: str):
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)

    return recognized, probability
def recognitionMontantEnLettre(model: Model):
    base_path = 'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/segments/'
    data={}
    print("Scanning ...")
    for infile in os.listdir(base_path):
        if infile.endswith('png'):
            r, p = infer(model, base_path+infile)
            data[r[0]] = p[0]
    print('recongnized : ', data)
    a=SpellingMistakeCorrection(data)
    print('corrected : ', a)
    return a


def num2words(num):
    under_20 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven',
                'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    above_100 = {100: 'Hundred', 1000: 'Thousand', 100000: 'Lakhs', 10000000: 'Crores'}
    if num < 20:
        return under_20[num]
    if num < 100:
        return tens[num // 10 - 2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])
    pivot = max([key for key in above_100.keys() if key <= num])

    return num2words(num // pivot) + ' ' + above_100[pivot] + ('' if num % pivot == 0 else ' ' + num2words(num % pivot))

def comparaisonMontant(montant_chiff,montant_lett:dict):
  montant_chiffre=num2words(montant_chiff)
  montant_chiffre=montant_chiffre.lower()
  c=[]
  for key,value in montant_lett.items():
    c.append(key)
  montant_lettre=' '.join(c)
  montant_lettre=montant_lettre.lower()
  if 'lakhs' in montant_chiffre:
    montant_chiffre=montant_chiffre.replace('lakhs','lakh')

  if montant_chiffre == montant_lettre :
    print('cheque correcte')
    return 1
  else:
    print('cheque à verifier')
    return montant_chiffre,montant_lettre

@app.route("/recognition", methods=['GET'])
def recognition():
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}

    decoder_type = decoder_mapping['bestpath']
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump='store_true')
    for lettre in os.listdir(LETTRE):
        if not "ipynb_checkpoints" in lettre:
            segmentationImage(r'C:/Users/yassi/Desktop/SimpleHTR-master/flaskApp/static/cheque/lettre/' + lettre)
            montant_lett = recognitionMontantEnLettre(model)
            deleteSegment()
    c = []
    for key, value in montant_lett.items():
        c.append(key)
    montant_lettre = ' '.join(c)
    montant_lettre = montant_lettre.lower()
    flash(montant_lettre)
    test = lists[0]
    montant_lettre_chiffre = num2words(int(test))
    a=comparaisonMontant(Montant_chiffre,montant_lett)
    print(a)
    return render_template('resultat.html',comparaison=a, chiffre = test,chiffre_lettre=montant_lettre_chiffre,lettre=montant_lettre)

@app.route("/resultat", methods=['GET'])
def resultat():
    return render_template('resultat.html')

############################################## MONTANT CHIFFRE #################################################


if __name__ == '__main__':
    app.run(debug=True)
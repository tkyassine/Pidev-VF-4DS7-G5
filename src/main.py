import argparse
import difflib
import json
from typing import Tuple, List
import editdistance
from path import Path

from spellchecker import SpellChecker
from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from segmentation import detectionPage, detectionWord, sort_words
from cheque_processor import *

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_number_list = '../model/NumberList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def number_list_from_file() -> List[str]:
    with open(FilePaths.fn_number_list) as f:
        return list(f.read())


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: str):
    """Recognizes text in image provided by file path."""
    test = {}
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)

    return recognized, probability

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

def recognitionMontantEnLettre(model: Model):
    base_path = '../data/Segments/'
    data={}
    print("Scanning ...")
    for infile in os.listdir(base_path):
        if infile.endswith('png'):
            r, p = infer(model, base_path+infile)
            data[r[0]] = p[0]
    print('recongnized : ',data)
    a=SpellingMistakeCorrection(data)
    print('corrected : ',a)

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
            save.save(r'../data/Segments/segment' + str(i) + '.png')
            i += 1

def deleteSegment():
    base_path = '../data/Segments/'
    for infile in os.listdir(base_path):
        if infile.endswith('png') and infile.startswith('segment'):
            os.remove(base_path+infile)

def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def SegmenterMontantEnLettre(model: Model):
    montantEnlettrePath = '../cheque/MontantEnLettre/'
    Segments = '../data/Segments/'

    if not os.path.exists(Segments):
        os.mkdir(Segments)
        print("Directory ", Segments, " Created ")
    else:
        print("Directory ", Segments, " already exists")

    for infile in os.listdir(montantEnlettrePath):
        print('file : ' + infile)
        segmentationImage(r'../cheque/MontantEnLettre/'+infile)
        recognitionMontantEnLettre(model)
        deleteSegment()

def inverserCouleurMontantEnChiffre():
    montantEnChiffrePath = '../cheque/MontantEnChiffre/'
    for infile in os.listdir(montantEnChiffrePath):
        print('file : ' + infile)
        im = cv2.imread(montantEnChiffrePath+infile)
        graychiffre = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(graychiffre, 150, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(montantEnChiffrePath + infile, opening)

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()

def all_the_process(model: Model):
    ConvertingToPNG()
    imagePreprocessing()
    repartirChequeSelonBanque()
    ExtraireMontantEnChiffre()
    inverserCouleurMontantEnChiffre()
    imagePreprocessing2()
    repartirChequeSelonBanque()
    ExtraireMontantEnLettre()
    SegmenterMontantEnLettre(model)

def main():
    """Main function."""
    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # when in line mode, take care to have a whitespace in the char list
        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters and words
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        all_the_process(model)
if __name__ == '__main__':
    main()

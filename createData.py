
from PIL import Image
from torchvision import transforms
import os
from sklearn import preprocessing
import pickle
import cv2

letter_map = {
'ain':'ain', 'alef':'alef', 'ayin':'ain', 'bet':'bet', 'dalet':'dalet', 'gimel':'gimel', 'he':'he', 'kaf':'kaf',
       'kaph':'kaf', 'khet':'khet', 'lamed':'lamed', 'mem':'mem', 'nun':'nun', 'pe':'pe', 'qof':'qof', 'qoph':'qof', 'resh':'resh',
       'reš':'resh', 'sameh':'sameh', 'samekh':'sameh', 'shin':'shin', 'tav':'tav', 'taw':'tav', 'tet':'tet', 'tsade':'sade',
       'vav':'vav', 'waw':'vav', 'yod':'yod', 'zayin':'zayin', 'šin':'shin', 'ḥet':'ḥet', 'ṣade':'sade', 'ṭet':'tet'}
DATA_PATH = '/home/olya/Documents/thesis/data'
def create_data_set( root ):
    letters = set()
    data = []

    for file in os.listdir(root):
        name = file.split('.', 1)[0].split('(',1)[0]
        name = letter_map[name]
        letters.add(name)

        gray = cv2.imread(os.path.join(root, file) , cv2.IMREAD_GRAYSCALE)

        # resize the images and invert it (black background)
        image = cv2.resize(255 - gray, (28, 28)) / 255
        image = Image.fromarray(image)
        data.append( (image, name ) )


    le = preprocessing.LabelEncoder()
    le.fit( [ label for _, label in data] )
    data = [ (le.transform([label])[0], image) for image, label in data ]
    return data, le



if __name__ == '__main__':
    #data, le = create_data_set('/home/olya/Documents/thesis/all')
    #pickle.dump( {'data': data, 'le':le}, open( DATA_PATH ,'wb'))
    with open( DATA_PATH,'rb') as f:
        le = pickle.load( f )['le']








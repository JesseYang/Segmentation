import argparse
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

from tensorpack import *

from train import Model
from cfgs.config import cfg
from predict import predict

def get_imglist(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

def predict(args):
    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["softmax_output"])
    predict_func = OfflinePredictor(predict_config)
    output_dir = args.output or "output"
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
    for (dirpath, dirnames, filenames) in os.walk(args.input):
        logger.info("Number of images to predict is " + str(len(filenames)) + ".")
        for file_idx, filename in enumerate(filenames):
            if file_idx % 10 == 0 and file_idx > 0:
                logger.info(str(file_idx) + "/" + str(len(filenames)))
            filepath = os.path.join(args.input, filename)
            newPredict_one(filepath, predict_func, os.path.join(output_dir, filename), args.crf)

#def predict(img_paths)

if __name__ == '__main__':
    test_list = get_imglist('test.txt')
    for i in test_list:
        img_name = i.split('/')[-1]
        img_name = img_name[:img_name.find('.jpg')]
        print(img_name)
    print(test_list)
    train_list = get_imglist('train.txt')
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help='path to the model file', required=True)
    #args = parser.parse_args()
    #predict(args, input = 'data/images')
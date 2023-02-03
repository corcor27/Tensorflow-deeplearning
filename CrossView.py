# from keras_cv_attention_models import coatnet
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from CustomDataGen import CustomDataGen
from Model_Class import CustomModel
import argparse
import sys
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score

"""
to run use for example:
python 5Clust_Train_CustModel.py --root_path "DATASET" --excel_file "DCIS_LIST2_numpy.xlsx" --wieghts_output_dir "WIEGHTS" --wieghts_path "None" --mode train
"""
def create_folds(No_folds, fold, data):
    train_val = []
    test = []
    scale = int(round(data.shape[0]/5))
    for ii in range(0, data.shape[0]):
        if fold*scale < ii <= (fold +1)*scale:
            test.append(ii)
        else:
            train_val.append(ii)
    split = int(round(len(train_val)*0.8,0))
    train, val = train_val[:split], train_val[split:]
    train_data = data.iloc[train]
    val_data = data.iloc[val]
    test_data = data.iloc[test]
    
    return train_data, val_data, test_data


improtant_classes = ['MARKID1','MARKID2', 'DCIS', 'INV']


parser = argparse.ArgumentParser()

###model options
parser.add_argument('--model_type',type=str, default = "Classification", help='classification or segmentation')
parser.add_argument('--fold', type=int,default=0, help='Training fold')
parser.add_argument('--mode',type=str, default="train", help='train or testing')
parser.add_argument('--connection_type', default = "Average" , help='do you want to use colour')
parser.add_argument('--pretrain', default = False , help='do you want to use a Resnet model')
parser.add_argument('--Num_folds', type=int, default = 5 , help='testing folds')
parser.add_argument('--repeat', type=int, default = 3, help='repeat counter')
###path selections
parser.add_argument('--dataset_path', type=str, default='DATASET/Evenscale_segs_bb_only', help='dataset_path')
parser.add_argument('--excel_file', type=str, help='list dir', default="INV_DCIS_DUAL_VIEW2_numpy2.xlsx")
parser.add_argument('--wieghts_output_dir', type=str, help='output dir', default="WIEGHTS")
parser.add_argument('--weight_path', type= str, default="None",help='weights path')

##training parmeters

parser.add_argument('--num_output_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--num_input_objects', type=int, default=2, help='number of inputs for network')
parser.add_argument('--Optimizer', type=str, default="Adam", help='number of inputs for network')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,default=256, help='input patch size of network input')
parser.add_argument('--aug', default = True, help='do you want augmentation? true or false')
parser.add_argument('--shuffle', default = True, help='shuffle ?')
parser.add_argument('--image_format',type=str, default = "numpy", help='image or numpy files')
parser.add_argument('--img_colour', default = False , help='do you want to use colour')
parser.add_argument('--loss_type', type=str, default = "Standard", help='repeat counter')

###Learning rate

parser.add_argument('--Max_lr', type=float,  default=0.005,help='network learning rate')
parser.add_argument('--Min_lr', type=float,  default=0.00000001,help='network learning rate')
parser.add_argument('--Starting_lr', type=float,  default=0.0001,help='network learning rate')
parser.add_argument('--lr_schedule', type= str, default="Cosine",help='weights path')
parser.add_argument('--warmup_epochs', type=int, default = 10, help='repeat counter')

###General parmeters

parser.add_argument('--seed', type=int,default=1234, help='random seed')

###Excel Parmeters

parser.add_argument('--image_col',type=str, default = "MARKID1", help='column name in excel that detail image names')
parser.add_argument('--second_image_col',type=str, default = "MARKID2", help='if dual input add 2nd col name')
parser.add_argument('-n' , '--prediction_classes', nargs='+', default =["DCIS" ,"INV"])

###Model Saving
parser.add_argument('--model_save', type=str, default = False, help='repeat counter')
parser.add_argument('--early_finish', type=str, default = True, help='repeat counter')
parser.add_argument('--post_analysis_folder', type=str, default = "Cross_analysis" , help='where to save the analysis')
parser.add_argument('--testing_metric', type=str, default = "Standard" , help='where to save the analysis')
parser.add_argument('--load_pretain', type=str, default = True , help='where to save the analysis')



args = parser.parse_args()
print(args)

if os.path.exists(args.post_analysis_folder) == False:
    os.mkdir(args.post_analysis_folder)
Fold_out_analysis = os.path.join(args.post_analysis_folder, str(args.fold))
if os.path.exists(Fold_out_analysis) == False:
    os.mkdir(Fold_out_analysis)



data = pd.read_excel(args.excel_file)

#data = data[improtant_classes]
if args.connection_type == "Multihead":
    args.batch_size = 16
if args.mode == "train":
    
    p = "{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size)
    args.wieghts_output_dir = os.path.join("WIEGHTS", p)
    
    train_data, val_data, test_data = create_folds(args.Num_folds, args.fold, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(val_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    my_model.Train_model()
    
elif args.mode == "inference":
    p = "{}_{}_{}_{}.h5".format(args.connection_type, args.fold, args.repeat, args.img_size)
    args.weight_path = os.path.join("WIEGHTS", p)
    print(args.weight_path)
    print('data.shape in loader = ', data.shape)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data, val_data, test_data = create_folds(args.Num_folds, args.fold, data)
    print(train_data.shape, val_data.shape, test_data.shape)
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_generator = CustomDataGen(train_data, args, mode = "training")
    train_generator.pre_check_images()
    
    print('Total data len = ', train_generator.n)
    
    print('Class Count = ', train_generator.Class_Count())
    valid_generator = CustomDataGen(val_data, args, mode = "validation")
    valid_generator.pre_check_images()
    print('Total val len = ', valid_generator.n)
    print('Class Count = ', valid_generator.Class_Count())
    args.batch_size = 1
    args.weight_path = os.path.join("WIEGHTS", p)
    my_model = CustomModel(args)
    my_model.Prep_training(train_generator, valid_generator)
    print('Class weights = ', my_model.Class_weights)
    Acc = my_model.predict_model()
    file_p = os.path.join(args.post_analysis_folder, "Run_Record.txt")
    if os.path.exists(file_p) == False:
        file1 = open(file_p, "w")
        L = ["Record of runs starting 25 Jan \n model:{} got ACC:{} on fold {}, image size:{}".format(args.connection_type, Acc, args.fold, args.img_size)]
        file1.writelines(L)
        file1.close()
    else:
        file1 = open(file_p, "a")  # append mode
        file1.write("\n model:{} got ACC:{} on fold {}, image size:{}".format(args.connection_type, Acc, args.fold, args.img_size))
        file1.close()
        

    
elif args.mode == "fold_testing":
    Ground = []
    prediction = []
    scale = int(round(data.shape[0]/5))
    for fod in range(0, args.Num_folds):
        test_data = data.iloc[scale*fod:scale*(fod+1)]
        args.batch_size = 1
        p = "{}_{}_{}_{}.h5".format(args.connection_type, fod, args.repeat, args.img_size)
        args.weight_path = os.path.join("WIEGHTS", p)
        valid_generator = CustomDataGen(test_data, args, mode = "validation")
        valid_generator.pre_check_images()
        print('Total val len = ', valid_generator.n)
        my_model = CustomModel(args)
        my_model.Prep_training(valid_generator, valid_generator)
        print('Class weights = ', my_model.Class_weights)
        Val_GRON, Val_PRED = my_model.predict_model_fold()
        Ground.append(Val_GRON), prediction.append(Val_PRED)
    Ground = np.concatenate(Ground)
    prediction = np.concatenate(prediction)
    ACC = my_model.predict_model_analysis(Ground, prediction)
    print(ACC)
    
        
        
    
    

    


from unicodedata import name
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import Dense, AveragePooling2D, Dropout, Flatten
from tensorflow_examples.models.pix2pix import pix2pix
import time
from CustomDataGen import CustomDataGen
import random
from datetime import datetime
import os
import cv2 
import math
import logging
from sklearn.utils import class_weight
from Model_Optimizer import CosineAnnealingScheduler # CustomSchedule
from tensorflow.keras.utils import Progbar
#from models_development import  DualResNet50, DualAttResNet50, Crossview2DualAttResNet50
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import sklearn.metrics as skm
from sklearn.metrics import accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Models import Create_Model

formatter = logging.Formatter('%(asctime)s - (%(name)s) %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    

class CustomModel:
    def __init__(self, args):
        self.args = args
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        if self.args.load_pretain == False:
            self.model = self.Create_Model_From_args()
        else:
            self.model = self.Create_Model_From_args()
            self.Load_model_weight()
        
    def Create_Model_From_args(self):
        if self.args.model_type == 'Classification':
            model = Create_Model(self.args).Create_Classification_Model()
            print(model)
        elif self.args.model_type == 'Segmentation':
            model = Create_Model(self.args)
        return model
        
    def Load_model_weight(self):
        if self.args.weight_path == 'None':
            #self.model.load_weights(self.weight_path)
            print("wieghts loading skipped")
        else:
            print("Wieghts loaded from {}".format(self.args.weight_path))
            self.model.load_weights(self.args.weight_path)
            
        
        
    def Save_model_weight(self):
        if self.args.wieghts_output_dir == 'None':
            print("No Weights path specified")
        else:
            if self.args.model_save:
                self.model.save(self.args.wieghts_output_dir,save_format='h5')
            else:
                self.model.save_weights(self.args.wieghts_output_dir)
                print("Wieghts are definetly saved")
    def define_optimizer(self):
        if self.args.Optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.args.Starting_lr)
        elif self.args.Optimizer == "SGD":
            return tf.keras.optimizers.experimental.SGD(learning_rate=self.args.Starting_lr)
    
    def loss_function(self):
        if self.args.model_type == 'Classification':
            if self.args.loss_type == "Standard":
                if self.args.num_output_classes < 3:
                    return keras.losses.CategoricalCrossentropy()
                else:
                    return keras.losses.BinaryCrossentropy()
    def Load_Lr_Schedule(self):
        if self.args.lr_schedule == "Cosine":
            return CosineAnnealingScheduler(self.args) #T_max=self.args.Max_lr, eta_max=5e-3, warmup_epochs = 10, eta_min=1e-7
        elif self.args.lr_schedule == "Exp_decay":
            self.reduce_rl_plateau = CustomSchedule() #patience=2,factor=0.00001,verbose=1, optim_lr=self.optimizer.learning_rate, reduce_lin=True
    
    def Get_testing_metric(self):
        if self.args.model_type == 'Classification':
            if self.args.testing_metric == "Accuracy":
                if self.args.num_output_classes < 3:
                    metric = keras.metrics.CategoricalAccuracy()
                    return metric
                else:
                    metric = keras.metrics.Accuracy()
                    return metric
    
    def Prep_training(self, Train_data, Validation_data):    
        self.optimizer = self.define_optimizer()
        self.loss_fn = self.loss_function()
        self.Schedule = self.Load_Lr_Schedule()
        self.train_acc_metric = keras.metrics.BinaryAccuracy()
        self.val_acc_metric = keras.metrics.BinaryAccuracy()
        self.Train_data = Train_data
        self.Validation_data = Validation_data
        self.Train_acc = np.zeros(shape=(self.args.max_epochs,), dtype=np.float32)
        self.Val_acc = np.zeros(shape=(self.args.max_epochs,), dtype=np.float32)
        self.model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.train_acc_metric)
        c_count = self.Train_data.Class_Count()
        self.Class_weights = c_count
        fac_c = 1 * np.sum(c_count)
        for ind in range(len(c_count)):
            score = math.log(fac_c/float(c_count[ind]))
            self.Class_weights[ind] = score if score > 1.0 else 1.0
            #self.Class_weights = np.sum(c_count) / c_count
            #self.Class_weights = self.Class_weights / np.sum(self.Class_weights)
            

    @tf.function
    def train_step(self, x, y, z, class_weights = None):
        with tf.GradientTape() as tape:
            y_pred = self.model([x,z], training=True)
            loss_value = self.loss_fn(y, y_pred)#, sample_weight=class_weights
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, y_pred)
        return loss_value, y_pred

    @tf.function
    def test_step(self, x, y, z):
        y_pred = self.model([x,z], training=False)
        loss_value = self.loss_fn(y, y_pred)
        self.val_acc_metric.update_state(y, y_pred)
        return loss_value

    @tf.function
    def __predict(self, x, z):
        val_pred = self.model([x,z], training=False)
        return val_pred

    def predict(self, img_path):
        if os.path.exists(img_path):
            img, img_org, org_height, org_width = self.Train_data.get_image(img_path)     
            val_pred = self.__predict(img)
            if self.model_type == 'Classification':
                return val_pred
            else:
                kernel_C = np.ones((9,9),np.uint8)
                pred_labels = np.squeeze(val_pred)
                out_image_mask = np.argmax(pred_labels, axis=2)
                out_image_mask[out_image_mask > 0] = 1
                out_image_mask = out_image_mask.astype(np.uint8)
                out_image_org = out_image_mask.copy()
                out_image_mask = cv2.morphologyEx(out_image_mask, cv2.MORPH_DILATE, kernel_C)
                #out_image_mask = cv2.morphologyEx(out_image_mask, cv2.MORPH_CLOSE, kernel_C)
                #out_image_mask = cv2.morphologyEx(out_image_mask, cv2.MORPH_OPEN, kernel_O)

                contours, hierarchy = cv2.findContours(out_image_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                try:
                    hierarchy = hierarchy[0]
                except:
                    hierarchy = []

                if len(contours) == 0:
                    return ''

                height, width = out_image_mask.shape
                min_x, min_y = width, height
                max_x = max_y = 0
                max_area = 0

                sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
                largest_item= sorted_contours[0]
                
                Mass_cont = cv2.moments(largest_item)
                cx = int(Mass_cont['m10']/Mass_cont['m00'])
                cy = int(Mass_cont['m01']/Mass_cont['m00'])

                (x, y, w, h) = cv2.boundingRect(largest_item)
                min_x, max_x = x, x + w
                min_y, max_y = y, y + h
                if max_x - min_x > 0 and max_y - min_y > 0:
                    min_x = max(1, min_x - math.floor(min_x*0.2))
                    max_x = min(width-1, max_x + math.floor(max_x*0.2))

                    min_y = max(1, min_y - math.floor(min_y*0.2))
                    max_y = min(height-1, max_y + math.floor(max_y*0.2))

                    x_len = max_x - min_x
                    y_len = max_y - min_y    
                    min_len = min(x_len, y_len)
                    if (min_len / x_len) > 0.8 and (min_len / y_len) > 0.8:
                        min_x_full = math.floor(((min_x-1) / self.image_size) * org_width)
                        max_x_full = math.floor(((max_x+1) / self.image_size) * org_width)
                        min_y_full = math.floor(((min_y-1) / self.image_size) * org_height)
                        max_y_full = math.floor(((max_y+1) / self.image_size) * org_height)
                        crop_img = img_org[min_y_full:max_y_full, min_x_full:max_x_full]
                    else:
                        min_2 = math.floor(min_len / 2)
                        height_2 = math.floor(height / 2) 
                        x_2 = math.floor((min_x + max_x) / 2) 
                        y_2 = math.floor((min_y + max_y) / 2) 
                        
                        min_x_full = math.floor(((x_2 - min_2-1) / self.image_size) * org_width)
                        max_x_full = math.floor(((x_2 + min_2+1) / self.image_size) * org_width)
                        min_y_full = math.floor(((y_2 - min_2-1) / self.image_size) * org_height)
                        max_y_full = math.floor(((y_2 + min_2+1) / self.image_size) * org_height)
                        crop_img = img_org[min_y_full:max_y_full, min_x_full:max_x_full]
                else:
                    crop_img = img_org
                    
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                return crop_img
    

    def plot_confusion(self, Val_PRED, Val_GRON):
        f, axes = plt.subplots(1, 2, figsize=(25, 15))
        axes = axes.ravel()
        for i in range(0, len(self.args.prediction_classes)):
            disp = ConfusionMatrixDisplay(confusion_matrix(Val_GRON[:, i], Val_PRED[:, i]),display_labels=[0, 1])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title('{}'.format(self.args.prediction_classes[i]))
            if i<10:
                disp.ax_.set_xlabel('Predition Label')
            if i%2!=0:
                disp.ax_.set_ylabel('True Label')
            disp.im_.colorbar.remove()
        
        plt.subplots_adjust(wspace=0.30, hspace=0.1)
        plt.rcParams['font.size'] = 60
        plt.suptitle("Model:{} Fold:{} Image Size:{}".format(self.args.connection_type, self.args.fold, self.args.img_size))
        f.colorbar(disp.im_, ax=axes)
        
        p = os.path.join(self.args.post_analysis_folder, str(self.args.fold), "{}_{}_{}_{}.png".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size))
        plt.savefig(p)
            
        return None

    def predict_model(self):
        Val_PRED = []
        Val_GRON = []
        for x_batch_val, z_batch_val, y_batch_val in self.Validation_data:
            val_pred = np.array(self.__predict(x_batch_val,  z_batch_val))
            y_pred = self.__predict(x_batch_val,  z_batch_val)
            self.val_acc_metric.update_state(y_batch_val, y_pred)
            POS = np.argmax(val_pred)
            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))
            
            prediction_array[:,POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)
           
        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))
        
        print(Val_PRED.shape, Val_GRON.shape)
        CON_MATRIX = skm.multilabel_confusion_matrix(Val_GRON, Val_PRED)
        print(CON_MATRIX)
        self.plot_confusion(Val_PRED, Val_GRON)
        Class_report = skm.classification_report(Val_GRON, Val_PRED, target_names=self.args.prediction_classes)
        file_p = os.path.join(self.args.post_analysis_folder, str(self.args.fold), "{}_{}_{}_{}.txt".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")

        file1.writelines(Class_report)
        file1.close()
        ACC = accuracy_score(Val_GRON, Val_PRED)
        print(ACC)
        ACC2 = keras.metrics.BinaryAccuracy()
        ACC2.update_state(Val_GRON, Val_PRED)
        ACC_SCORE = ACC2.result().numpy()
        print(ACC_SCORE)
        
        m = self.val_acc_metric.result()
        print(m)

        return ACC
    
    def predict_model_analysis(self, Val_GRON, Val_PRED):
        CON_MATRIX = skm.multilabel_confusion_matrix(Val_GRON, Val_PRED)
        print(CON_MATRIX)
        self.plot_confusion(Val_PRED, Val_GRON)
        Class_report = skm.classification_report(Val_GRON, Val_PRED, target_names=self.args.prediction_classes)
        file_p = os.path.join(self.args.post_analysis_folder, "{}_{}.txt".format(self.args.connection_type, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")
        file1.writelines(Class_report)
        file1.close()
        ACC = accuracy_score(Val_GRON, Val_PRED)
        
        print(ACC)
        return ACC
    
    def predict_model_fold(self):
        Val_PRED = []
        Val_GRON = []
        for x_batch_val, z_batch_val, y_batch_val in self.Validation_data:
            val_pred = np.array(self.__predict(x_batch_val,  z_batch_val))
            POS = np.argmax(val_pred)
            #prediction = tf.math.argmax(val_pred, axis=0, output_type=tf.int64)
            #y_batch_val = tf.math.argmax(y_batch_val, axis=0, output_type=tf.int64)
            prediction_array = np.zeros((y_batch_val.shape))
            
            prediction_array[:,POS] = 1
            Val_PRED.append(prediction_array)
            Val_GRON.append(y_batch_val)
           
        Val_PRED = np.squeeze(np.array(Val_PRED))
        Val_GRON = np.squeeze(np.array(Val_GRON))
        ACC = accuracy_score(Val_GRON, Val_PRED)
        print(ACC)
        return Val_GRON, Val_PRED
    
    def create_training_logs(self, training, validation, Final_epochs):
        file_p = os.path.join(self.args.post_analysis_folder, str(self.args.fold), "Training_logs_{}_{}_{}_{}.txt".format(self.args.connection_type, self.args.fold, self.args.repeat, self.args.img_size))
        print(file_p)
        file1 = open(file_p, "w")
        report_list = []
        for epoch in range(0, Final_epochs):
            report_list.append("Training_loss: {}, Validation_loss:{} \n".format(training[epoch], validation[epoch]))
        report = "".join(report_list)
        file1.writelines(report)
        file1.close()
        
        
        return None
    
    def Train_model(self):
        count = 0
        Training_loss = []
        Validation_loss = []
        
        epo = []
        metrics_names = ['train_loss'] 
        for epoch in range(self.args.max_epochs):
            epo.append(epoch)
            validation_acc = []
            self.Schedule.on_epoch_begin(epoch, self.model.optimizer)
            start_time = time.time()
            pb_i = Progbar(self.Train_data.n, stateful_metrics=metrics_names)
            # Iterate over the batches of the dataset.
            train_loss_value = []
            for step, (x_batch_train, z_batch_train, y_batch_train) in enumerate(self.Train_data):
                if random.uniform(0, 1) < (0.4/(epoch+1)) and self.args.model_type == 'Classification' and self.Train_data.augmentation:
                    random_number = random.randint(1, 8)
                    x_batch_train, z_batch_train = self.Train_data.batch_augmentation(x_batch_train, random_number), self.Train_data.batch_augmentation(z_batch_train, random_number)
                if self.args.model_type == 'Segmentation':
                    class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(np.ravel(y_batch_train,order='C')), y = np.ravel(y_batch_train,order='C'))
                    class_weights = dict(zip(np.unique(y_batch_train), class_weights))
                    f_class_weights = np.zeros([6])
                    for key in class_weights:    
                        f_class_weights[int(key)] = class_weights[key]
                    sample_weights = np.take(np.array(f_class_weights), np.round(y_batch_train).astype('int'))
                    #class_weights = tf.constant(sample_weights)
                    loss_value, y_pred = self.train_step(x_batch_train, y_batch_train, sample_weights)
                else:
                    sample_weights = np.zeros(self.args.batch_size)
                    for ind in range(self.args.batch_size):
                        CLASS_LIST = np.argmax(y_batch_train[ind,:])
                        sample_weights[ind] = self.Class_weights[CLASS_LIST]
                    #tf.constant(self.Class_weights) #np.repeat(self.Class_weights[np.newaxis,...], self.Batch_size, axis=0) 
                    loss_value, y_pred = self.train_step(x_batch_train, y_batch_train, z_batch_train, sample_weights)
                    train_loss_value.append(loss_value)
                values=[('train_loss', loss_value)]
                pb_i.add(self.args.batch_size, values=values)
            # Display metrics at the end of each epoch.
            Training_loss.append(np.mean(train_loss_value))
            self.Train_acc[epoch] = self.train_acc_metric.result()
            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            Validation_loss_value = []
            for x_batch_val, z_batch_val, y_batch_val in self.Validation_data:
                loss_value = self.test_step(x_batch_val, y_batch_val, z_batch_val)
                Validation_loss_value.append(loss_value)
            Validation_loss.append(np.mean(Validation_loss_value))    
            self.Val_acc[epoch] = self.val_acc_metric.result()
            
            #self.reduce_rl_plateau.on_epoch_end(epoch, self.Val_acc[epoch])
            self.Schedule.on_epoch_end(epoch, self.model.optimizer)
            print(self.Val_acc[epoch])
            if np.argmax(self.Val_acc) == epoch or epoch == 0:
                self.Save_model_weight()
                print("Weights Saved")
                count = 0
                threshold = 30
                cut_off = 5
                print("End Count: {} Cut off at: {}".format(count, cut_off))
            else:
                print("End Count: {} Cut off at: {}".format(count, cut_off))
                if epoch > threshold:
                    count += 1      

            self.Train_data.on_epoch_end()
            self.val_acc_metric.reset_states()
            if count == cut_off:
                print("breaking loop")
                break
        self.create_training_logs(Training_loss, Validation_loss, max(epo))
        

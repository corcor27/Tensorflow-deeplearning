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

class Create_Model():
    def __init__(self, args):
        self.args = args

    def Averaged_Excitation_Attention(self, x,y):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16, activation='relu')(x_att)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)
        
        y_att = GlobalAveragePooling2D()(y)
        y_d = Dense(y_att.shape[1]/16, activation='relu')(y_att)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)
        
        Combined_att = tf.keras.layers.Average()([x_d, y_d])
        
        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])
        return x_att, y_att
    
    def Self_Excitation_Attention(self, x,y):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16, activation='relu')(x_att)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)
        
        y_att = GlobalAveragePooling2D()(y)
        y_d = Dense(y_att.shape[1]/16, activation='relu')(y_att)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)
        
        x_att = tf.keras.layers.Multiply()([x, x_d])
        y_att = tf.keras.layers.Multiply()([y, y_d])
        return x_att, y_att
    
    def Addition_Excitation_Attention(self, x,y):
        x_att = GlobalAveragePooling2D()(x)
        x_d = Dense(x_att.shape[1]/16, activation='relu')(x_att)
        x_d = Dense(x_att.shape[1], activation='sigmoid')(x_d)
        
        y_att = GlobalAveragePooling2D()(y)
        y_d = Dense(y_att.shape[1]/16, activation='relu')(y_att)
        y_d = Dense(y_att.shape[1], activation='sigmoid')(y_d)
        
        Combined_att = Add()([x_d, y_d])
        
        x_att = tf.keras.layers.Multiply()([x, Combined_att])
        y_att = tf.keras.layers.Multiply()([y, Combined_att])
        return x_att, y_att
    
    def Pretrained_Model(self):
        baseA = tf.keras.applications.resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=(self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        baseB = tf.keras.applications.resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=(self.args.img_size, self.args.img_size, self.N_channels), classes=1000)
        for layer in baseA.layers:
            layer._name = layer.name + str("_1")
        for layer in baseA.weights:
            layer._name = layer.name + str("_1")
        POOLA = MaxPooling2D((7, 7))(baseA.output)
        POOLB = MaxPooling2D((7, 7))(baseB.output)
        fc2 = self.Output_Block(POOLA,POOLB)
        model = Model(inputs=[baseA.input, baseB.input], outputs=fc2)
        return model
    
    def Output_Block(self, x,y):
        combined = concatenate([x, y])
        dropout = Dropout(0.3)(combined)
        flattened = Flatten()(dropout)
        fc1 = Dense(224, activation="relu")(flattened)  # was 100
        fc2 = Dense(self.args.num_output_classes, activation="softmax")(fc1)
        return fc2
    
    def Create_Classification_Model(self):
        if self.args.pretrain == True:
            model = self.Pretrained_Model()
        else:
            model = self.ResNet50()
        return model
    
    def identity_block4(self, input_tensorA, input_tensorB, kernel_size, filters):
        nb_filter1, nb_filter2, nb_filter3 = filters
        
        MLO = self.side_stack(input_tensorA, nb_filter1, nb_filter2, nb_filter3, kernel_size)
        CC = self.side_stack(input_tensorB, nb_filter1, nb_filter2, nb_filter3, kernel_size)
        
        MLO = Add()([MLO, input_tensorA])
        MLO = Activation('relu')(MLO)
        CC = Add()([CC, input_tensorB])
        CC = Activation('relu')(CC)
        
        return MLO, CC
    def conv_block4(self, input_tensorA, input_tensorB, kernel_size, filters, strides=(2, 2)):
        
        nb_filter1, nb_filter2, nb_filter3 = filters
        
        MLO = self.side_stack2(input_tensorA, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides = strides)
        CC = self.side_stack2(input_tensorB, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides = strides)
        
        MLO_shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensorA)
        MLO_shortcut = BatchNormalization(momentum=0.1,epsilon=0.00001)(MLO_shortcut)
        
        CC_shortcut = Conv2D(nb_filter3, (1, 1), strides=strides)(input_tensorB)
        CC_shortcut = BatchNormalization(momentum=0.1,epsilon=0.00001)(CC_shortcut)
    
        MLO = Add()([MLO, MLO_shortcut])
        MLO = Activation('relu')(MLO)
        
        CC = Add()([CC, CC_shortcut])
        CC = Activation('relu')(CC)
        return MLO, CC
    
    def side_stack(self, x, nb_filter1, nb_filter2, nb_filter3, kernel_size):
        x = Conv2D(nb_filter1, (1, 1))(x)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
        x = Activation('relu')(x)
    
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter3, (1, 1))(x)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
        return x
    
    def side_stack2(self,input_tensor, nb_filter1, nb_filter2, nb_filter3, kernel_size, strides=(2, 2)):
        x = Conv2D(nb_filter1, (1, 1), strides=strides)(input_tensor)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
        x = Activation('relu')(x)
    
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(nb_filter2, (kernel_size, kernel_size))(x)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter3, (1, 1))(x)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    
    
        return x
    def MultiheadAttentionDirectSum(self, x, y):
        x,y = self.crossview_component(x,y)
        return x,y
    
    def DualResNet50_TOP(self,MLO, CC, filters=64):
        MLO, CC = self.conv_block4(MLO, CC, 3, [filters, filters, 4*filters], strides=(1, 1))
        
        MLO, CC = self.identity_block4(MLO, CC, 3, [filters, filters, 4*filters])
        MLO, CC = self.identity_block4(MLO, CC, 3, [filters, filters, 4*filters])
        MLO, CC = self.conv_block4(MLO, CC, 3, [2*filters, 2*filters, 8*filters])
        
        for i in range(0, 3):
            MLO, CC = self.identity_block4(MLO, CC, 3, [2*filters, 2*filters, 8*filters])
        
        return MLO,CC
    
    def augment(self, x):
        #x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
        #x = tf.keras.layers.RandomRotation(0.2)(x)
        return x
    
    
    
    def Input_Block(self,inputA, inputB):
        
        #AugA = self.augment(inputA)
        #AugB = self.augment(inputB)
        MLO = self.input_block(inputA)
        CC = self.input_block(inputB)
        return MLO, CC
    
    def input_block(self,x, filters=32):
        x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(filters, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((7, 7), strides=(2, 2))(x)
        return x
    
    def DualResNet50_BOT(self,MLO, CC, filters=64):
        
    
        MLO, CC = self.conv_block4(MLO, CC, 3, [4*filters, 4*filters, 16*filters])
        
        
        for i in range(0, 5):
            MLO, CC = self.identity_block4(MLO, CC, 3, [4*filters, 4*filters, 16*filters])
    
        MLO, CC = self.conv_block4(MLO, CC, 3, [8*filters, 8*filters, 32*filters])
        
        
        MLO, CC = self.identity_block4(MLO, CC, 3, [8*filters, 8*filters, 32*filters])
        MLO, CC = self.identity_block4(MLO, CC, 3, [8*filters, 8*filters, 32*filters]) #2048
        
        MLO = MaxPooling2D((7, 7))(MLO)
        CC = MaxPooling2D((7, 7))(CC)
        return MLO,CC
        
    def ResNet50(self):
        if self.args.img_colour:
            self.N_channels = 3
        else:
            self.N_channels = 1
        inputA = Input(shape=(self.args.img_size, self.args.img_size, self.N_channels))
        inputB = Input(shape=(self.args.img_size, self.args.img_size, self.N_channels))
        InputA, InputB = self.Input_Block(inputA, inputB)
        ResTopA, ResTopB = self.DualResNet50_TOP(InputA, InputB)
        if self.args.connection_type == "Average":
            CrossA, CrossB = self.Averaged_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Self":
            CrossA, CrossB = self.Self_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Add":
            CrossA, CrossB = self.Addition_Excitation_Attention(ResTopA, ResTopB)
        elif self.args.connection_type == "Multihead":
            CrossA, CrossB = self.MultiheadAttentionDirectSum(ResTopA, ResTopB)
        elif self.args.connection_type == "Baseline":
            CrossA, CrossB = ResTopA, ResTopB
        ResBotA, ResBotB = self.DualResNet50_BOT(CrossA, CrossB)
        Out  = self.Output_Block(ResBotA, ResBotB)
        
        model = Model(inputs=[inputA, inputB], outputs=Out)
        return model
        
    
    def Create_Segmentation_Model(self):
        base = keras.applications.EfficientNetB7(input_shape=[self.image_size, self.image_size, self.N_channels], include_top=False, weights=None)
            
        base.load_weights("WIEGHTS/256_eff_base.h5")
    
    
        skip_names = ['block1a_activation', # size 64*64
                      'block2g_activation',  # size 32*32
                      'block3g_activation',  # size 16*16
                      'block5g_activation',  'top_activation']
    
        skip_outputs = [base.get_layer(name).output for name in skip_names]
    
        downstack = keras.Model(inputs=base.input, outputs=skip_outputs)
        downstack.trainable = False
    
        # Four upstack blocks for upsampling sizes
        # 4->8, 8->16, 16->32, 32->64
        upstack = [pix2pix.upsample(512,self.N_channels),
                   pix2pix.upsample(256,self.N_channels),
                   pix2pix.upsample(128,self.N_channels),
                   pix2pix.upsample(64,self.N_channels)]
        
        #We can explore the individual layers in each upstack block.
        #upstack[0].layers
        
        inputs = keras.layers.Input(shape=[self.image_size, self.image_size, self.N_channels])
        
        # downsample
        down = downstack(inputs)
        out = down[-1]
        # prepare skip-connections
        skips = reversed(down[:-1])
        # choose the last layer at first 4 --> 8
        # upsample with skip-connections
        for up, skip in zip(upstack,skips):
            out = up(out)
            out = keras.layers.Concatenate()([out,skip])
        # define the final transpose conv layer
        # image 128 by 128 with 59 classes
        out = keras.layers.Conv2DTranspose(self.No_output + 1, self.N_channels, strides=2, padding='same', )(out)
         # complete unet model
        unet = keras.Model(inputs=inputs, outputs=out)
        return unet
    
    def tokenizer(self, x, tokens):
        flatten_A = tf.reshape(x, [-1, x.shape[1]* x.shape[2], x.shape[3]])
        embedA = tf.keras.layers.Conv1D(tokens, kernel_size = 1)(flatten_A)
        soft = tf.nn.softmax(embedA, axis=1)
        matrixmult_overbatch = tf.matmul(flatten_A, soft)
        embed2 = tf.keras.layers.Conv1D(tokens, kernel_size = 1)(matrixmult_overbatch)
        reorder = tf.transpose(embed2, [0, 2, 1])
        map_token = Dense(reorder.shape[2],activation=None)(reorder)
        reorder2 = tf.transpose(map_token, [0, 2, 1])
        reorder_flattened = tf.transpose(flatten_A, [0, 2, 1])
        
        matrixmult_overbatch2 = tf.matmul(map_token, reorder_flattened)
        reorder_matrixmult_overbatch2 = tf.transpose(matrixmult_overbatch2, [0, 2, 1])
        soft2 = tf.nn.softmax(reorder_matrixmult_overbatch2, axis=1)
        matrixmult_overbatch3 = tf.matmul(reorder_flattened, soft2)
        
        
        return matrixmult_overbatch3, soft2
    
    def embed_tensor(self,x, embedding=32, heads=12):
        
        embed2 = tf.keras.layers.Conv1D(embedding*heads, kernel_size = 1)(x)
        
        return embed2
    
    def reverse_tokenizer(self, y, att):
        tran_att = tf.transpose(att, [0, 2, 1])
        matrixmult_overbatch = tf.matmul(y, tran_att)
        
        
        return matrixmult_overbatch
    
    def scaled_dot_product(self,q,k,v, i):
        #calculates Q . K(transpose)
        Q = q[:,i,:,:]
        K = k[:,i,:,:]
        reorderQ = tf.transpose(Q, [0, 2, 1])
        qkt = tf.matmul(reorderQ,K)
        #caculates scaling factor
        dk = tf.math.sqrt(tf.cast(reorderQ.shape[-1],dtype=tf.float32))
        scaled_qkt = qkt/dk
        softmax = tf.nn.softmax(scaled_qkt,axis=-1)
        
        z = tf.matmul(softmax,v)
        #z = tf.reshape(z, [-1, 1, z.shape[1], z.shape[2]])
        #shape: (m,Tx,depth), same shape as q,k,v
        return z
    
    def in_add_linear(self, x,y, drop):
        conx = tf.keras.layers.Conv1D(x.shape[-1], kernel_size = 1)(x)
        dropoutx = Dropout(drop)(conx)
        addition = dropoutx + y
        norm = BatchNormalization(momentum=0.1,epsilon=0.00001)(addition)
        return norm
    
    def multiheadatt(self, Q, K, V, heads, emb):
        multi_attn = []
        for i in range(heads):
            multi_attn.append(self.scaled_dot_product(Q,K,V, i))
        multi_head = tf.concat(multi_attn,axis=1)
        reorder_mult_head = tf.transpose(multi_head, [0, 2, 1])
        multi_head_attention = Dense(V.shape[1])(reorder_mult_head)
        return multi_head_attention
        
    def crossview_component(self, x,y, tokenizer_a = False,tokenizer_b = False, num_heads = 12, embedding = 32, dropout = 0.1):
        tokensA = x
        tokensB = y
        if tokenizer_a == True:
            tokensA, attA = self.tokenizer(x, tokens)
            reorder2 = tf.transpose(tokensA, [0, 2, 1])
            embed_A = self.embed_tensor(reorder2, heads = num_heads)
            
        else:
            flatten_A = tf.reshape(tokensA, [-1, tokensA.shape[1]* tokensA.shape[2], tokensA.shape[3]])
            embed_A = self.embed_tensor(flatten_A, heads = num_heads)
            
        
        if tokenizer_b == True:
            tokensB, attB = self.tokenizer(y, tokens)
            reorderB = tf.transpose(tokensB, [0, 2, 1])
            embed_B = self.embed_tensor(reorderB, heads = num_heads)
        else:
            flatten_B = tf.reshape(tokensB, [-1, tokensB.shape[1]* tokensB.shape[2], tokensB.shape[3]])
            embed_B = self.embed_tensor(flatten_B, heads = num_heads)
            
    
        reshape_embeddedQ = tf.reshape(embed_A, [-1, embed_A.shape[1], num_heads, embedding])
        reorderQ = tf.transpose(reshape_embeddedQ, [0, 2, 3,1])
        reshape_embeddedK = tf.reshape(embed_B, [-1, embed_B.shape[1], num_heads, embedding])
        reorderK = tf.transpose(reshape_embeddedK, [0, 2, 3,1])
        flattenVB = tf.reshape(y, [-1, y.shape[1]* y.shape[2], y.shape[3]])
        flattenVA = tf.reshape(x, [-1, x.shape[1]* x.shape[2], x.shape[3]])
        #reorderV = tf.transpose(FlattenV, [0, 2, 1])
        multA = self.multiheadatt(reorderQ, reorderK, flattenVB, num_heads, embedding)
        multB= self.multiheadatt(reorderK, reorderQ, flattenVA, num_heads, embedding)
        reordermultA = tf.transpose(multA, [0, 2, 1])
        reshapemultA = tf.reshape(reordermultA, [-1, x.shape[1], x.shape[2], x.shape[3]])
        linearA_addition = self.in_add_linear(reshapemultA,y, dropout)
        
        reordermultB = tf.transpose(multB, [0, 2, 1])
        reshapemultB = tf.reshape(reordermultB, [-1, y.shape[1], y.shape[2], y.shape[3]])
        linearB_addition = self.in_add_linear(reshapemultB,x, dropout)
        
        #reshape_V = tf.reshape(tokensB, [-1,num_heads, tokensB.shape[-1], tokensA.shape[-1]])
        #multi = MultiHeadAttention(d_model=reshape_embeddedA.shape[-1], num_heads=num_heads)
        #out, attn = multi(reshape_embeddedA, k=reshape_embeddedB, q=embed_B, mask=None)
        
        #revserse_embed_a = reverse_tokenizer(embedA, attA)
        return linearA_addition, linearB_addition
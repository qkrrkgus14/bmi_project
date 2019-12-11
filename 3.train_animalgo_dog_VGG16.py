# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import load_miracle_VGG16 as lm
from keras.layers import *
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import VGG19,Xception ,ResNet50, inception_v3, MobileNet



from keras.backend import tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


categories = ["affenpinscher","afghanhound","airedaleterrier","akita","alaskanmalamute","american_bully","americancockerspaniel","americanstaffordshireterrier","australiancattledog","australiankelpie",
              "australianshepherd","australiansilkyterrier","australianterrier","basenji","bassethound","beagle","beardedcollie","bedlingtonterrier","belgiangriffon","belgianshepherddoggroenedael",
              "belgianshepherddoglaekenois","belgianshepherddogmalinois","belgianshepherddogtervueren","bergerdebeauce","bernerhound","bernesemountaindog","bichonfrise","bloodhound","bolognese","bordeauxmastiff",
              "bordercollie","borderterrier","borzoi","bostonterrier","bouvierdesflanders","brazilianguarddog","briard","brittanyspaniel","brusselsgriffon","bulldog","bullmastiff","bullterrier","cairnterrier","cavalierkingcharlesspaniel",
              "chesapeakebayretriever","chihuahua","chin","chinesecresteddog","chowchow","clumberspaniel","cotondetulear","curlycoatedretriever","dachshund","dalmatian","dandiedinmontterrier","deerhound","dobermann","dogoargentino",
              "englishcockerspaniel","englishpointer","englishsetter","englishspringerspaniel","estrelamountaindog","fieldspaniel","flatcoatedretriever","frenchbulldog","germanboxer","germanhuntingterrier","germanshepherddog",
              "germanshorthairedpointer","germanwirehairedpointer","giantschnauzer","goldenretriever","gordonsetter","greatdane","greatjapanesedog","greyhound","hokkaido","hungarianshorthairedvizsla","ibizanhound","irishredandwhitesetter",
              "irishredsetter","irishsoftcoatedwheatenterrier","irishterrier","irishwolfhound","italiangreyhound","jackrussellterrier","japanesespitz","japaneseterrier","kai","keeshond","kerryblueterrier","kingcharlesspaniel","kishu","komondor",
              "kooikerhondje","koreajindodog","kuvasz","labradorretriever","lakelandterrier","largemunsterlander","leonberger","lhasaapso","lowchen","maltese","maltipoo","manchesterterrier","maremmaandabruzzessheepdog","mastiff","mexicanhairlessdog",
              "miniaturebullterrier","miniaturepinscher","miniatureschnauzer","neapolitanmastiff","newfoundland","norfolkterrier","norwegianbuhund","norwegianelkhoundgrey","norwichterrier","novascotiaducktollingretriever","oldenglishsheepdog",
              "papillon","parsonrussellterrier","pekingese","peruvianhairlessdog","petitbassetgriffonvendeen","petitbrabancon","pharaohhound","polishlowlandsheepdog","pomeranian","poodle","portuguesewaterdog","pug","puli","pumi","pyreneanmastiff",
              "pyreneanmountaindog","pyreneansheepdog","rhodesianridgeback","rottweiler","roughcollie","saintbernarddog","saluki","samoyed","schipperke","scottishterrier","sealyhamterrier","sharpei","shetlandsheepdog","shiba","shihtzu","shikoku",
              "siberianhusky","skyeterrier","sloughi","smoothcollie","smoothfoxterrier","spanishmastiff","staffordshirebullterrier","standardschnauzer","thairidgebackdog","tibetanmastiff","tibetanspaniel","tibetanterrier","tosa",
              "toymanchesterterrier","weimaraner","welshcorgicardigan","welshcorgipembroke","welshspringerspaniel","welshterrier","westhighlandwhiteterrier","whippet","whiteswissshepherddog","wirefoxterrier","yorkshireterrier"]

nb_classes=len(categories)

for d in ['/gpu:0']:
    with tf.device(d):
        x_train, x_test, t_train, t_test=lm.image_2cha_dog()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255
        x_test /= 255

        print(x_train.shape)
        print(x_test.shape)


        #1
        model=Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=x_train.shape[1:]))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2),strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        #2
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2),strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        #3
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2),strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        '''
        #4
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2),strides=(2,2)))
        #5
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2),strides=(2,2)))
        '''
        
    
        model.add(GlobalAveragePooling2D())
        #model.add(Flatten())
        

        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.summary()

        #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        sgd = SGD(lr=0.001, decay = 1e-7, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])#,options=run_opts)

        
        callbacks_list = [
            EarlyStopping(             
                monitor="val_acc",
                patience=9,    
            ),
            ModelCheckpoint(
                filepath="VGG16_2Cha_Dog_CP.h5py",
                monitor="val_loss",
                save_best_only=False,  
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,            
                patience=7,
            ),
        ]
        
history = model.fit(x_train, t_train, batch_size=32, epochs=25, verbose=1, validation_split=0.2, callbacks=callbacks_list)

score=model.evaluate(x_test,t_test, verbose=1)

print('loss=',score[0])
print('accuracy=',score[1])

animalgo10_params = "./model_VGG16_2Cha_dog.h5py" #저장경로
model.save_weights(animalgo10_params)   #저장


plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.title('2Cha_Mobile_Dog')
plt.ylabel('Acc')
plt.xlabel('Apochs')
plt.legend(['loss', 'acc'], loc='upper left')
plt.show()

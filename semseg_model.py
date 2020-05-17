from utils.data_generator import ImageDataGenerator
from utils.helpers import color_encode
from utils.utils import load_image, decode_one_hot
from keras_applications import imagenet_utils

from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.losses import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils import utils
from builders import builder
import tensorflow as tf
import argparse
import os

from dataset import ImageMaskDataset



class SemanticSegmentationModel:
    MD_FCN8 = 'FCN-8s'
    MD_FCN16 = 'FCN-16s'
    MD_FCN32 = 'FCN-32s'
    MD_UNet = 'UNet'
    MD_SegNet = 'SegNet'
    MD_BayesSegNet = 'Bayesian-SegNet'
    MD_PAN = 'PAN'
    MD_PSPNet = 'PSPNet'
    MD_RefineNet = 'RefineNet'
    MD_DenseASPP = 'DenseASPP'
    MD_DeepLabV3 = 'DeepLabV3'
    MD_DeepLabV3Plus = 'DeepLabV3Plus'
    MD_BiSegNet = 'BiSegNet'



    def __init__(self, model_path, model, n_labels, input_size, base_model = None):
        self.model_path = model_path
        self.dir_logs = os.path.join(self.model_path, "logs")
        self.dir_weights = os.path.join(self.model_path, "weights")
        self.dir_checkpoints = os.path.join(self.model_path, "checkpoints")

        self.n_labels = n_labels
        self.model_name = model
        self.base_model_name = base_model
        self.input_size = input_size

        mmodel, bm = builder(self.n_labels, self.input_size, self.model_name, self.base_model_name)
        self.model = mmodel
        self.base_model = bm
        self.model.summary()

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.isdir(self.dir_logs):
            os.mkdir(self.dir_logs)
        if not os.path.isdir(self.dir_weights):
            os.mkdir(self.dir_weights)
        if not os.path.isdir(self.dir_checkpoints):
            os.mkdir(self.dir_checkpoints)

    def train(self, dataset:ImageMaskDataset,
                    loss = "ce",
                    random_crop = False, 
                    crop_height = 256,
                    crop_width = 256,
                    batch_size = 5,
                    valid_batch_size = 1,
                    num_epochs = 100,
                    initial_epoch = 0,
                    h_flip = False,
                    v_flip = False,
                    brightness = None,
                    rotation = 0.0,
                    zoom_range = 0.0,
                    channel_shift = 0.0,
                    data_aug_rate = 0.0,
                    checkpoint_freq = 1,
                    validation_freq = 1,
                    num_valid_images = 20,
                    data_shuffle = True,
                    random_seed = None,
                    weights = None,
                    steps_per_epoch = 1000,
                    lr_scheduler = 'cosine_decay',
                    lr_warmup=False, 
                    learning_rate =3e-4,
                    optimizer='adam'):
        """
        
        :param dataset:
        :param loss: ['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss']
        :param num_classes: 
        :param random_crop: 
        :param crop_height: 
        :param crop_width: 
        :param batch_size: 
        :param valid_batch_size: 
        :param num_epochs: 
        :param initial_epoch: 
        :param h_flip: 
        :param v_flip: 
        :param brightness: 
        :param rotation: 
        :param zoom_range: 
        :param channel_shift: 
        :param data_aug_rate: 
        :param checkpoint_freq: 
        :param validation_freq: 
        :param num_valid_images: 
        :param data_shuffle: 
        :param random_seed: 
        :param weights: 
        :param steps_per_epoch: 
        :param lr_scheduler: ['step_decay', 'poly_decay', 'cosine_decay']
        :param lr_warmup: 
        :param learning_rate: 
        :param optimizer: ['sgd', 'adam', 'nadam', 'adamw', 'nadamw', 'sgdw']
        :return: 
        """

        train, test = dataset.split()

        # summary
        self.model.summary()

        # load weights
        if weights is not None:
            print('Loading the weights...')
            self.model.load_weights(weights)

        # chose loss
        losses = {'ce': categorical_crossentropy_with_logits,
                  'focal_loss': focal_loss(),
                  'miou_loss': miou_loss(num_classes=self.n_labels),
                  'self_balanced_focal_loss': self_balanced_focal_loss()}
        loss = losses[loss] if loss is not None else categorical_crossentropy_with_logits

        # chose optimizer
        total_iterations = len(train) * num_epochs // batch_size
        wd_dict = utils.get_weight_decays(self.model)
        ordered_values = []
        weight_decays = utils.fill_dict_in_order(wd_dict, ordered_values)

        optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      'nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                      'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99),
                      'adamw': AdamW(learning_rate=learning_rate, batch_size=batch_size,
                                     total_iterations=total_iterations),
                      'nadamw': NadamW(learning_rate=learning_rate, batch_size=batch_size,
                                       total_iterations=total_iterations),
                      'sgdw': SGDW(learning_rate=learning_rate, momentum=0.99, batch_size=batch_size,
                                   total_iterations=total_iterations)}

        # lr schedule strategy
        if lr_warmup and num_epochs - 5 <= 0:
            raise ValueError('num_epochs must be larger than 5 if lr warm up is used.')

        lr_decays = {
            'step_decay': step_decay(learning_rate, num_epochs - 5 if lr_warmup else num_epochs,
                                     warmup=lr_warmup),
            'poly_decay': poly_decay(learning_rate, num_epochs - 5 if lr_warmup else num_epochs,
                                     warmup=lr_warmup),
            'cosine_decay': cosine_decay(num_epochs - 5 if lr_warmup else num_epochs,
                                         learning_rate, warmup=lr_warmup)}
        lr_decay = lr_decays[lr_scheduler]

        # training and validation steps
        steps_per_epoch = len(
            train) // batch_size if not steps_per_epoch else steps_per_epoch
        validation_steps = num_valid_images // valid_batch_size

        # compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="categorical_crossentropy",
                    metrics=[MeanIoU(self.n_labels), "accuracy"])
        # data generator
        # data augmentation setting
        train_gen = ImageDataGenerator(random_crop=random_crop,
                                       rotation_range=rotation,
                                       brightness_range=brightness,
                                       zoom_range=zoom_range,
                                       channel_shift_range=channel_shift,
                                       horizontal_flip=v_flip,
                                       vertical_flip=v_flip)

        valid_gen = ImageDataGenerator()

        train_generator = train_gen.flow(images_list=train.values,
                                         labels_list=train.labels,
                                         num_classes=self.n_labels,
                                         batch_size=batch_size,
                                         target_size=self.input_size,
                                         shuffle=data_shuffle,
                                         seed=random_seed,
                                         data_aug_rate=data_aug_rate)

        valid_generator = valid_gen.flow(images_list=test.values,
                                         labels_list=test.labels,
                                         num_classes=self.n_labels,
                                         batch_size=valid_batch_size,
                                         target_size=self.input_size)

        # callbacks setting
        # checkpoint setting
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.dir_checkpoints,
                                  '{model}_based_on_{base}_'.format(model=self.model_name, base=self.base_model_name) +
                                  'miou_{val_mean_io_u:04f}_' + 'ep_{epoch:02d}.h5'),
            save_best_only=True, period=checkpoint_freq, monitor='val_mean_io_u', mode='max')
        # tensorboard setting
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.dir_logs)
        es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        # learning rate scheduler setting
        learning_rate_scheduler = LearningRateScheduler(lr_decay, learning_rate, lr_warmup, steps_per_epoch,
                                                        verbose=1)

        callbacks = [model_checkpoint, tensorboard, learning_rate_scheduler, es_cb]

        # training...
        self.model.fit_generator(train_generator,
                          steps_per_epoch=steps_per_epoch,
                          epochs=num_epochs,
                          callbacks=callbacks,
                          validation_data=valid_generator,
                          validation_steps=validation_steps,
                          validation_freq=validation_freq,
                          max_queue_size=10,
                          workers=os.cpu_count(),
                          use_multiprocessing=False,
                          initial_epoch=initial_epoch)

        # save weights
        self.model.save(filepath=os.path.join(self.dir_weights, '{model}_based_on_{base_model}.h5'.format(model=self.model_name, base_model=self.base_model_name)))

    def load_fit(self, p):
        X = load_image(p)
        return self.fit(X)

    def fit(self, X):
        image = cv2.resize(X, dsize=self.input_size)
        image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')

        # image processing
        if np.ndim(image) == 3:
            image = np.expand_dims(image, axis=0)
        assert np.ndim(image) == 4

        # get the prediction
        prediction = self.model.predict(image)

        if np.ndim(prediction) == 4:
            prediction = np.squeeze(prediction, axis=0)

        # decode one-hot
        prediction = decode_one_hot(prediction)

        # color encode
        if color_encode:
            prediction = color_encode(prediction, color_values)

        return prediction

import cv2
ds = ImageMaskDataset().load("lip.json")
# for img, ms in ds:
#     img = cv2.imread(img)
#     if len(img.shape) == 2:
#         print("SHIT")
model = SemanticSegmentationModel("output", SemanticSegmentationModel.MD_DeepLabV3Plus, 20, (512,512))
model.train(ds, batch_size=2, steps_per_epoch=1000)
model.predict(my_image)
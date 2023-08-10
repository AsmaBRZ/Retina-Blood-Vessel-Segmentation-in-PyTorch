import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
from PIL import Image
from collections import OrderedDict
import math
import tensorflow_addons as tfa

def load_images(images_dir):
    images = {}
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            file_image = os.path.join(root,file)
            name = int(file.split(".")[0].split('_')[0])
            image = tf.keras.utils.img_to_array(Image.open(file_image))
            images[name] = image
    return OrderedDict(sorted(images.items()))

def random_shift(img, mask,annotations, shift = 0.02):        
        # Compute the maximum shift in pixels
        h, w, _ = img.shape
        h_shift = int(h * shift)
        w_shift = int(w * shift)

        # Generate random shift values
        h_shift_val = tf.random.uniform(shape=[], minval=-h_shift, maxval=h_shift, dtype=tf.int32)
        w_shift_val = tf.random.uniform(shape=[], minval=-w_shift, maxval=w_shift, dtype=tf.int32)

        # Apply the shift to the image
        shifted_img = tf.roll(img, shift=(h_shift_val, w_shift_val), axis=(0, 1))
        shifted_mask = tf.roll(mask, shift=(h_shift_val, w_shift_val), axis=(0, 1))
        shifted_annotations = tf.roll(annotations, shift=(h_shift_val, w_shift_val), axis=(0, 1))

        return shifted_img, shifted_mask, shifted_annotations

base_dir ='./dataset'
train_images_dir = os.path.join(base_dir, 'training/images/')
train_masks_dir = os.path.join(base_dir, 'training/mask')
train_annotations_dir = os.path.join(base_dir, 'training/1st_manual')
test_images_dir = os.path.join(base_dir, 'test/images')
test_masks_dir = os.path.join(base_dir, 'test/mask')

new_base_dir  = "./new_dataset"
new_train_dir = os.path.join(new_base_dir, 'training/')
new_test_dir = os.path.join(new_base_dir, 'test/')
new_train_images_dir = os.path.join(new_base_dir, 'training/images/')
new_train_masks_dir = os.path.join(new_base_dir, 'training/mask/')
new_train_annotations_dir = os.path.join(new_base_dir, 'training/1st_manual/')
new_test_images_dir = os.path.join(new_base_dir, 'test/images/')
new_test_masks_dir = os.path.join(new_base_dir, 'test/mask/')


if not os.path.exists(new_base_dir):
    os.mkdir(new_base_dir)
if not os.path.exists(new_train_dir):
    os.mkdir(new_train_dir)
if not os.path.exists(new_test_dir):
    os.mkdir(new_test_dir)
if not os.path.exists(new_train_images_dir):  
    os.mkdir(new_train_images_dir)
if not os.path.exists(new_train_masks_dir):    
    os.mkdir(new_train_masks_dir)
if not os.path.exists(new_train_annotations_dir):
    os.mkdir(new_train_annotations_dir)
if not os.path.exists(new_test_images_dir):
    os.mkdir(new_test_images_dir)
if not os.path.exists(new_test_masks_dir):
    os.mkdir(new_test_masks_dir)

#create a list to store images, masks and annotations
train_images = load_images(train_images_dir)
train_masks = load_images(train_masks_dir)
train_annotations = load_images(train_annotations_dir)
test_images = load_images(test_images_dir)
test_masks = load_images(test_masks_dir)


for k, v in test_images.items():
    img = v
    mask = test_masks[k]
    tf.io.write_file(new_test_images_dir+str(k)+"_test.png", tf.io.encode_png(tf.cast(img, dtype=tf.uint8)))
    tf.io.write_file(new_test_masks_dir+str(k)+"_test_mask.png", tf.io.encode_png(tf.cast(mask, dtype=tf.uint8)))

#shift
index = 0
for k, v in train_images.items():
    img = v
    mask = train_masks[k]
    annotations = train_annotations[k]
    shifted_img, shifted_mask, shifted_annotations = random_shift(img, mask,annotations)
    tf.io.write_file(new_train_images_dir+str(index)+"_training.png", tf.io.encode_png(tf.cast(img, dtype=tf.uint8)))
    tf.io.write_file(new_train_masks_dir+str(index)+"_training_mask.png", tf.io.encode_png(tf.cast(mask, dtype=tf.uint8)))
    tf.io.write_file(new_train_annotations_dir+str(index)+"_manual1.png", tf.io.encode_png(tf.cast(annotations, dtype=tf.uint8)))
    index += 1
    tf.io.write_file(new_train_images_dir+str(index)+"_training.png", tf.io.encode_png(tf.cast(shifted_img, dtype=tf.uint8)))
    tf.io.write_file(new_train_masks_dir+str(index)+"_training_mask.png", tf.io.encode_png(tf.cast(shifted_mask, dtype=tf.uint8)))
    tf.io.write_file(new_train_annotations_dir+str(index)+"_manual1.png", tf.io.encode_png(tf.cast(shifted_annotations, dtype=tf.uint8)))
    index += 1


#gamma
train_images = load_images(new_train_images_dir)
train_masks = load_images(new_train_masks_dir)
train_annotations = load_images(new_train_annotations_dir)
for k, v in train_images.items():
    img = v
    mask = train_masks[k]
    annotations = train_annotations[k]
    bright_img = tf.image.adjust_gamma(img, 0.9)
    tf.io.write_file(new_train_images_dir+str(index)+"_training.png", tf.io.encode_png(tf.cast(bright_img, dtype=tf.uint8)))
    tf.io.write_file(new_train_masks_dir+str(index)+"_training_mask.png", tf.io.encode_png(tf.cast(mask, dtype=tf.uint8)))
    tf.io.write_file(new_train_annotations_dir+str(index)+"_manual1.png", tf.io.encode_png(tf.cast(annotations, dtype=tf.uint8)))
    index += 1

"""
#flip RL
train_images = load_images(new_train_images_dir)
train_masks = load_images(new_train_masks_dir)
train_annotations = load_images(new_train_annotations_dir)

for k, v in train_images.items():
    img = v
    mask = train_masks[k]
    annotations = train_annotations[k]
    flip_LR_img = tf.image.flip_left_right(img)
    flip_LR_mask = tf.image.flip_left_right(mask)
    flip_LR_annotations = tf.image.flip_left_right(annotations)
    tf.io.write_file(new_train_images_dir+str(index)+"_training.png", tf.io.encode_png(tf.cast(flip_LR_img, dtype=tf.uint8)))
    tf.io.write_file(new_train_masks_dir+str(index)+"_training_mask.png", tf.io.encode_png(tf.cast(flip_LR_mask, dtype=tf.uint8)))
    tf.io.write_file(new_train_annotations_dir+str(index)+"_manual1.png", tf.io.encode_png(tf.cast(flip_LR_annotations, dtype=tf.uint8)))
    index += 1

#flip UP
train_images = load_images(new_train_images_dir)
train_masks = load_images(new_train_masks_dir)
train_annotations = load_images(new_train_annotations_dir)

for k, v in train_images.items():
    img = v
    mask = train_masks[k]
    annotations = train_annotations[k]
    flip_LR_img = tf.image.flip_up_down(img)
    flip_LR_mask = tf.image.flip_up_down(mask)
    flip_LR_annotations = tf.image.flip_up_down(annotations)
    tf.io.write_file(new_train_images_dir+str(index)+"_training.png", tf.io.encode_png(tf.cast(flip_LR_img, dtype=tf.uint8)))
    tf.io.write_file(new_train_masks_dir+str(index)+"_training_mask.png", tf.io.encode_png(tf.cast(flip_LR_mask, dtype=tf.uint8)))
    tf.io.write_file(new_train_annotations_dir+str(index)+"_manual1.png", tf.io.encode_png(tf.cast(flip_LR_annotations, dtype=tf.uint8)))
    index += 1

"""
#Rotate
train_images = load_images(new_train_images_dir)
train_masks = load_images(new_train_masks_dir)
train_annotations = load_images(new_train_annotations_dir)

angles = [30,60,90,120,150,180]
for k, v in train_images.items():
    img = v
    mask = train_masks[k]
    annotations = train_annotations[k]
    for angle in angles:
        img_rot = tfa.image.rotate(img, angles = angle * math.pi / 180, interpolation='BILINEAR')
        mask_rot = tfa.image.rotate(mask, angles = angle * math.pi / 180, interpolation='BILINEAR')
        annotations_rot = tfa.image.rotate(annotations, angles = angle * math.pi / 180, interpolation='NEAREST')
        tf.io.write_file(new_train_images_dir+str(index)+"_training.png", tf.io.encode_png(tf.cast(img_rot, dtype=tf.uint8)))
        tf.io.write_file(new_train_masks_dir+str(index)+"_training_mask.png", tf.io.encode_png(tf.cast(mask_rot, dtype=tf.uint8)))
        tf.io.write_file(new_train_annotations_dir+str(index)+"_manual1.png", tf.io.encode_png(tf.cast(annotations_rot, dtype=tf.uint8)))
        index += 1


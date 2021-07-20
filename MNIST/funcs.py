import tensorflow as tf


def shape_mine_img(img_path='', img_h=28, img_w=28):
    image = tf.io.read_file(img_path)
    train_images = np.asarray(_preprocess_image(image, img_h=img_h, img_w=img_w))
    train_images = train_images.reshape((-1, img_h * img_w))
    return train_images[0]


def _preprocess_image(image, img_h=28, img_w=28):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [img_w, img_h])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, [img_w, img_h])
    return image

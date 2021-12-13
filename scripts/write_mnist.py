import tensorflow.keras as tfk
import skimage.io
import pathlib
import pandas as pd

# data_dir = pathlib.Path('/home/vafaeisa/scratch/datasets')
data_dir = pathlib.Path('../../datasets')


def write_images(images, labels_list, img_dir):
    labels = {}
    for i, (img, l) in enumerate(zip(images, labels_list)):
        img_name = f'image_{i}.jpg'
        skimage.io.imsave(img_dir.joinpath(img_name), img)
        labels[img_name] = l

    names = list(labels.keys())
    labels = list(labels.values())

    pd.DataFrame({'image name': names, 'label': labels}).to_csv(img_dir.joinpath('labels.csv'))


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = tfk.datasets.mnist.load_data()

    train_images = train_images[:1000]
    test_images = test_images[:1000]

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    data_dir = data_dir.joinpath('mnist')
    data_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = data_dir.joinpath('train')
    train_img_dir.mkdir(exist_ok=True)
    write_images(train_images, train_labels, train_img_dir)

    val_img_dir = data_dir.joinpath('validation')
    val_img_dir.mkdir(exist_ok=True)
    write_images(test_images[:500], test_labels[:500], val_img_dir)

    test_img_dir = data_dir.joinpath('test')
    test_img_dir.mkdir(exist_ok=True)
    write_images(test_images[500:], test_labels[500:], test_img_dir)

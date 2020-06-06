import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config


def view_data():
    nrows, ncols = 3, 3
    pic_index = 0
    fig = plt.gcf()
    fig.set_size_inches(nrows * 4, ncols * 4)

    pic_index = 2
    next_rock = [os.path.join(config.TRAIN_DIR, 'rock', fname) for fname in os.listdir(os.path.join(config.TRAIN_DIR, 'rock'))[pic_index-2:pic_index]]
    next_paper = [os.path.join(config.TRAIN_DIR, 'paper', fname) for fname in os.listdir(os.path.join(config.TRAIN_DIR, 'paper'))[pic_index-2:pic_index]]
    next_scissors = [os.path.join(config.TRAIN_DIR, 'scissors', fname) for fname in os.listdir(os.path.join(config.TRAIN_DIR, 'scissors'))[pic_index-2:pic_index]]

    for i, img_path in enumerate(next_rock+next_paper+next_scissors):
        sp = plt.subplot(nrows, ncols, i+1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def plot_graphs(history, string):
    plt.plot(history[string])
    plt.plot(history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


if __name__ == '__main__':
    view_data()

    # history = np.load(f'{config.MODEL_PATH}my_history.npy', allow_pickle=True).item()
    # plot_graphs(history, 'accuracy')
    # plot_graphs(history, 'loss')

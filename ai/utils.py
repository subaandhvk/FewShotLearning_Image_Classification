import os, cv2, random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras.applications.mobilenet import preprocess_input
from ai.constants import IMG_WIDTH, IMG_HEIGHT, ROOT, TRAIN_SIZE, CFM_FILENAME_TRUE, CFM_FILENAME_FALSE

# rename files in path to numbers starting from 0
from ai.neural_nets import classify_images_2


def get_folders_list():
  return os.listdir(ROOT)


def rename_files(path):
    for i, filename in enumerate(os.listdir(path)):
        os.rename(path + "/" + filename, path + "/" + str(i) + ".jpg")


# Read an image and convert it to a numpy array
def read_image(index):
    path = os.path.join(ROOT, index[0], index[1])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dim = (IMG_WIDTH, IMG_HEIGHT)
    # resize image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image


# Split dataset into training and testing sets
def split_dataset(directory, split=TRAIN_SIZE):
    folders = os.listdir(directory)
    for fder in folders:
        rename_files(os.path.join(directory, fder))
    num_train = int(len(folders) * split)

    random.shuffle(folders)

    train_list, test_list = {}, {}

    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files

    # Creating Test-list
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files

    return train_list, test_list


def split_dataset_by_triplets(directory, split=TRAIN_SIZE):
    folders = os.listdir(directory)

    all_triplets = create_triplets(ROOT, dict(folders))

    num_train = int(len(all_triplets) * split)

    random.shuffle(all_triplets)

    train_list, test_list = [], []

    # Creating Train-list
    for triplet in all_triplets[:num_train]:
        train_list.append(triplet)

    # Creating Test-list
    for triplet in all_triplets[num_train:]:
        test_list.append(triplet)

    return train_list, test_list


def create_triplets(directory, folder_list, max_files=100):
    triplets = []
    folders = list(folder_list.keys())
    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)

        for i in range(num_files - 1):
            for j in range(i + 1, num_files):
                anchor = (folder, f"{i}.jpg")
                positive = (folder, f"{j}.jpg")
                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_file = random.randint(0, folder_list[neg_folder] - 1)
                negative = (neg_folder, f"{neg_file}.jpg")

                triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets


# Get batches of triplets
def get_batch(triplet_list, batch_size=256, preprocess=True, return_true_class=False):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor = []
        positive = []
        negative = []
        true_class_values = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(read_image(a))
            positive.append(read_image(p))
            negative.append(read_image(n))
            true_class_values = a
            j += 1

        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)
        if return_true_class:
            yield ([anchor, positive, negative, true_class_values])
        else:
            yield ([anchor, positive, negative])


def test_on_triplets(test_triplet, siamese_model, batch_size=256):
    pos_scores, neg_scores = [], []

    for data in get_batch(test_triplet, batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on test = {accuracy:.5f}")
    return (accuracy, ap_mean, an_mean, ap_stds, an_stds)


# Plot model metrics
def plot_metrics(loss, metrics):
    # Extracting individual metrics from metrics
    accuracy = metrics[:, 0]
    ap_mean = metrics[:, 1]
    an_mean = metrics[:, 2]
    ap_stds = metrics[:, 3]
    an_stds = metrics[:, 4]

    plt.figure(figsize=(15, 5))

    # Plotting the loss over epochs
    plt.subplot(121)
    plt.plot(loss, 'b', label='Loss')
    plt.title('Training loss')
    plt.legend()

    # Plotting the accuracy over epochs
    plt.subplot(122)
    plt.plot(accuracy, 'r', label='Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()

    plt.figure(figsize=(15, 5))

    # Comparing the Means over epochs
    plt.subplot(121)
    plt.plot(ap_mean, 'b', label='AP Mean')
    plt.plot(an_mean, 'g', label='AN Mean')
    plt.title('Means Comparision')
    plt.legend()

    # Plotting the accuracy
    ap_75quartile = (ap_mean + ap_stds)
    an_75quartile = (an_mean - an_stds)
    plt.subplot(122)
    plt.plot(ap_75quartile, 'b', label='AP (Mean+SD)')
    plt.plot(an_75quartile, 'g', label='AN (Mean-SD)')
    plt.title('75th Quartile Comparision')
    plt.legend()


def class_confusion_matrix(test_list, test_triplet):
    pred_pos_list = []
    pred_neg_list = []
    true_pos_list = []
    true_neg_list = []

    for data in get_batch(test_triplet, batch_size=256, return_true_class=True):
        a, p, n, pos_class_name, neg_class_name = data
        pos_classification = classify_images_2(a, p, pos_class_name, neg_class_name)
        neg_classification = classify_images_2(a, n, pos_class_name, neg_class_name)

        pred_pos_list += pos_classification
        pred_neg_list += neg_classification
        true_pos_list += pos_class_name
        true_neg_list += neg_class_name
        break

    cf_matrix_true = confusion_matrix(true_pos_list, pred_pos_list)
    cf_matrix_true = cf_matrix_true.astype('float') / cf_matrix_true.sum(axis=1)[:, np.newaxis]

    print(cf_matrix_true)

    cm_df_true = pd.DataFrame(cf_matrix_true)

    cf_matrix_false = confusion_matrix(true_neg_list, pred_neg_list)
    cf_matrix_false = cf_matrix_false.astype('float') / cf_matrix_false.sum(axis=1)[:, np.newaxis]

    print(cf_matrix_false)

    cm_df_false = pd.DataFrame(cf_matrix_false)

    classes = get_folders_list()
    # Plotting the positive confusion matrix
    plt.figure(figsize=(5, 4))
    cfm_plot_true = sns.heatmap(cm_df_true, annot=True)
    plt.title('Confusion Matrix for positive images')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=0)
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    cfm_plot_true.figure.savefig(CFM_FILENAME_TRUE)

    # Plotting the negative confusion matrix
    plt.figure(figsize=(5, 4))
    cfm_plot_true = sns.heatmap(cm_df_false, annot=True)
    plt.title('Confusion Matrix for negative images')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=0)
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')

    plt.show()
    cfm_plot_true.figure.savefig(CFM_FILENAME_FALSE)
    return [CFM_FILENAME_TRUE, CFM_FILENAME_FALSE]









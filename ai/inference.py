import os
import random
import statistics

import numpy as np
import tensorflow as tf

from ai.constants import RANDOM_SEED, ROOT, PREDICT_FOLDER, BATCH_SIZE, PREDICT_THRESHOLD, TRAIN_SIZE
from ai.neural_nets import get_siamese_network, SiameseModel, extract_encoder
from ai.train import train_model
from ai.utils import create_triplets, get_batch, class_confusion_matrix, split_dataset_by_triplets

from tensorflow.keras.optimizers import Adam


def create_and_train_model():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Split dataset into training and testing sets
    train_list, test_list = split_dataset_by_triplets(ROOT, split=TRAIN_SIZE)

    # Training and testing triplets
    train_triplet, test_triplet = split_dataset_by_triplets(ROOT, split=TRAIN_SIZE)

    # Get the siamese network
    siamese_network = get_siamese_network()
    model_summary = siamese_network.summary()

    # Siamese model
    siamese_model = SiameseModel(siamese_network)

    # Optimizer and model compilation
    optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
    siamese_model.compile(optimizer=optimizer)

    # Train model
    train_loss, test_metrics = train_model(train_triplet, test_triplet, siamese_model)

    encoder = extract_encoder(siamese_model)
    encoder.save_weights("encoder")
    encoder_summary = encoder.summary()

    # Save class confusion matrix to image
    class_confusion_matrix(test_list, test_triplet)

    return {"model_summary": model_summary, "train_loss": train_loss, "test_metrics": test_metrics, "encoder_summary": encoder_summary}


def predict_image_class(siamese_model):
    image_class = None
    print("in predict image")
    pos_scores, neg_scores, triplets = [], [], []
    folders = os.listdir(ROOT)
    for fder in folders:
        if fder != PREDICT_FOLDER:
            random_fder_img = os.listdir(os.path.join(ROOT, fder))
            for img in random_fder_img:
                n_t = (fder, img)
                triplets.append(tuple([('new', '0.jpg'), ('new', '1.jpg'), n_t]))

    # get triplet scores
    for data in get_batch(triplets, batch_size=BATCH_SIZE):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    triplet_dict = {}
    for idx in range(len(triplets)):
        triplet = triplets[idx]
        if triplet[2][0] not in triplet_dict:
            triplet_dict[triplet[2][0]] = [idx]
        else:
            triplet_dict[triplet[2][0]].append(idx)

    # compute sums of negatives
    negative_sums_dict = {}
    for k, v in triplet_dict.items():
        negative_sums_dict[k] = statistics.mean([neg_scores[i] for i in triplet_dict[k]])

    new_dict = dict((k, v) for k, v in negative_sums_dict.items() if v <= PREDICT_THRESHOLD)
    image_class = ', '.join(new_dict.keys())

    print("image_class", image_class)

    return image_class
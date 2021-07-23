import time

from ai.constants import BATCH_SIZE, EPOCHS, MAX_ACCURACY, FINAL_SIAMESE_MODEL_FILE, \
    SIAMESE_MODEL_FILE
from ai.utils import get_batch, test_on_triplets


def train_model(train_triplets, test_triplets, siamese_model):
    save_all = False
    train_loss = []
    test_metrics = []
    for epoch in range(1, EPOCHS + 1):
        t = time.time()

        # Training the model on train data
        epoch_loss = []
        for data in get_batch(train_triplets, batch_size=BATCH_SIZE, preprocess=True):
            loss = siamese_model.train_on_batch(data)
            epoch_loss.append(loss)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        train_loss.append(epoch_loss)

        print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
        print(f"Loss on train    = {epoch_loss:.5f}")

        # Testing the model on test data
        metric = test_on_triplets(test_triplets, siamese_model, batch_size=BATCH_SIZE)
        test_metrics.append(metric)
        accuracy = metric[0]

        # Saving the model weights
        if save_all or accuracy >= MAX_ACCURACY:
            siamese_model.save_weights(SIAMESE_MODEL_FILE)
            max_acc = accuracy
    # Saving the model after all epochs run
    siamese_model.save_weights(FINAL_SIAMESE_MODEL_FILE)
    return train_loss, test_metrics



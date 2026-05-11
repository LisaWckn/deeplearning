import keras
import tensorflow as tf
from keras import layers

class LoadDataSet:

    def __init__(self, dataset_dir, dataset_test_dir, normalize = False, augment = False, img_size=(224, 224), batch_size=32):
        self.dataset_dir = dataset_dir
        self.dataset_test_dir = dataset_test_dir
        self.normalize = normalize
        self.augment = augment
        self.img_size = img_size
        self.batch_size = batch_size

        self.loadDataset()
        self.loadTestDataset()

    def getTrainDataset(self):
        return self.train_dataset, self.val_dataset
    
    def getTestDataset(self):
        return self.test_dataset, self.class_names

    def loadDataset(self, validation_split=0.2, seed=123):
        full_dataset = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=validation_split,
            subset="both",
            seed=seed
        )
        self.train_class_names = full_dataset[0].class_names
        self.val_class_names = full_dataset[1].class_names

        train_dataset = full_dataset[0].cache().shuffle(1000)
        val_dataset = full_dataset[1].cache()

        if self.normalize:
            train_dataset = self.addNormalization(train_dataset)
            val_dataset = self.addNormalization(val_dataset)

        if self.augment:
            train_dataset = self.addAugmentation(train_dataset)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def loadTestDataset(self):
        self.dataset_test_dir = keras.utils.image_dataset_from_directory(
            self.dataset_test_dir,
            labels="inferred",
            label_mode="int",
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.class_names = self.dataset_test_dir.class_names
        dataset = self.dataset_test_dir.map(lambda x, y: (x, y))

        if self.normalize:
            dataset = self.addNormalization(dataset)
            
        self.test_dataset = dataset

    def check_label_consistency(self):
        train_labels = set(self.train_class_names)
        val_labels = set(self.val_class_names)
        test_labels = set(self.class_names)

        same_train_val = train_labels == val_labels
        same_train_test = train_labels == test_labels

        print("Train labels:", self.train_class_names)
        print("Val labels:  ", self.val_class_names)
        print("Test labels: ", self.class_names)

        if not same_train_val:
            print("Mismatch train/val:", sorted(train_labels.symmetric_difference(val_labels)))
        if not same_train_test:
            print("Mismatch train/test:", sorted(train_labels.symmetric_difference(test_labels)))

        return same_train_val and same_train_test

    def addAugmentation(self, dataset):
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.1),
        ])
        return dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


    def addNormalization(self, dataset):
        normalization = keras.Sequential([
            layers.Rescaling(1./255),
        ])
        return dataset.map(lambda x, y: (normalization(x), y))
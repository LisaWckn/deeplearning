import keras

class LoadDataSet:

    def __init__(self, dataset_dir, dataset_test_dir, img_size=(224, 224), batch_size=32):
        # Verzeichnisse und Parameter für Bildgröße und Batch-Größe speichern
        self.dataset_dir = dataset_dir
        self.dataset_test_dir = dataset_test_dir
        self.img_size = img_size
        self.batch_size = batch_size

        # Trainings-/Validierungsdaten und Testdaten direkt beim Erzeugen laden
        self.loadDataset()
        self.loadTestDataset()

    def getTrainDataset(self):
        # Gibt Trainings- und Validierungsdatensatz zurück
        return self.train_dataset, self.val_dataset
    
    def getTestDataset(self):
        # Gibt Testdatensatz und Klassennamen zurück
        return self.test_dataset, self.class_names

    def loadDataset(self, validation_split=0.2, seed=123):
        # Lädt Bilder aus dem Trainingsordner und teilt sie automatisch
        # in Trainings- und Validierungsdaten auf
        full_dataset = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            labels="inferred",          # Labels aus Ordnernamen ableiten
            label_mode="int",           # Klassen als Integer kodieren
            image_size=self.img_size,   # Bilder auf einheitliche Größe skalieren
            batch_size=self.batch_size,
            validation_split=validation_split,
            subset="both",              # Train- und Validation-Split gleichzeitig erzeugen
            seed=seed,
            shuffle=True
        )

        # Klassennamen aus beiden Datensätzen speichern
        self.train_class_names = full_dataset[0].class_names
        self.val_class_names = full_dataset[1].class_names

        # Cache verbessert die Performance beim wiederholten Laden
        # Shuffle nur beim Trainingsdatensatz, damit die Reihenfolge gemischt wird
        train_dataset = full_dataset[0].cache().shuffle(1000)
        val_dataset = full_dataset[1].cache()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def loadTestDataset(self):
        # Lädt den Testdatensatz separat
        # shuffle=False ist wichtig, damit die Reihenfolge für Auswertungen stabil bleibt
        self.dataset_test_dir = keras.utils.image_dataset_from_directory(
            self.dataset_test_dir,
            labels="inferred",
            label_mode="int",
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Klassennamen des Testdatensatzes speichern
        self.class_names = self.dataset_test_dir.class_names

        # Datensatz in (x, y)-Paare umwandeln
        # Die Map-Funktion verändert hier inhaltlich nichts,
        # sorgt aber für eine explizite Struktur
        dataset = self.dataset_test_dir.map(lambda x, y: (x, y))
            
        self.test_dataset = dataset

    def check_label_consistency(self):
        # Prüft, ob in Train, Validation und Test dieselben Klassen vorhanden sind
        train_labels = set(self.train_class_names)
        val_labels = set(self.val_class_names)
        test_labels = set(self.class_names)

        same_train_val = train_labels == val_labels
        same_train_test = train_labels == test_labels

        # Ausgabe der erkannten Klassen zum Debuggen
        print("Train labels:", self.train_class_names)
        print("Val labels:  ", self.val_class_names)
        print("Test labels: ", self.class_names)

        # Falls Unterschiede vorhanden sind, werden diese explizit angezeigt
        if not same_train_val:
            print("Mismatch train/val:", sorted(train_labels.symmetric_difference(val_labels)))
        if not same_train_test:
            print("Mismatch train/test:", sorted(train_labels.symmetric_difference(test_labels)))

        # True nur dann, wenn alle Label-Sets übereinstimmen
        return same_train_val and same_train_test
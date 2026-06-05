import numpy as np
import keras

class EvaluateModel:

    def evaluate_model(self, test_dataset, class_names, model_path):
        # Gespeichertes Modell laden
        # custom_objects ist nötig, falls beim Modellaufbau spezielle Funktionen
        # wie preprocess_input verwendet wurden
        model = keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": keras.applications.resnet_v2.preprocess_input
            }
        )

        # Gefundene Klassen zur Kontrolle ausgeben
        print("Klassen:", class_names)

        # Modell auf dem gesamten Testdatensatz auswerten
        # Gibt Verlustwert und Gesamtgenauigkeit zurück
        loss, accuracy = model.evaluate(test_dataset, verbose=1)
        print(f"\nGesamtgenauigkeit: {accuracy * 100:.2f}%")

        # Listen für echte Labels und vorhergesagte Labels vorbereiten
        y_true = []
        y_pred = []

        # Testdatensatz batchweise durchlaufen
        for images, labels in test_dataset:
            # Vorhersagen für den aktuellen Batch berechnen
            predictions = model.predict(images, verbose=0)

            # Aus den Wahrscheinlichkeiten die Klasse mit dem höchsten Wert auswählen
            predicted_labels = np.argmax(predictions, axis=-1)

            # Echte und vorhergesagte Labels sammeln
            y_true.append(labels.numpy())
            y_pred.append(predicted_labels)

        # Alle Batch-Ergebnisse zu je einem Array zusammenführen
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Zähler für Anzahl und korrekt klassifizierte Bilder pro Kategorie
        category_counts = {name: 0 for name in class_names}
        category_correct = {name: 0 for name in class_names}

        # Für jedes Testbeispiel prüfen, ob die Vorhersage korrekt war
        for true_label, pred_label in zip(y_true, y_pred):
            category_name = class_names[true_label]
            category_counts[category_name] += 1

            if true_label == pred_label:
                category_correct[category_name] += 1

        # Ausgabe der Genauigkeit pro Klasse/Kategorie
        print("\nGenauigkeit je Kategorie:")
        for category_name in class_names:
            count = category_counts[category_name]
            correct = category_correct[category_name]

            # Falls zu einer Klasse keine Testdaten vorhanden sind
            if count == 0:
                print(f"  {category_name}: keine Testdaten")
            else:
                print(f"  {category_name}: {correct}/{count} = {correct / count * 100:.2f}%")
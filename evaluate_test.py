import numpy as np
import keras

class EvaluateModel:

    def evaluate_model(self, test_dataset, class_names, model_path):
        model = keras.models.load_model(model_path)

        print("Klassen:", class_names)

        loss, accuracy = model.evaluate(test_dataset, verbose=1)
        print(f"\nGesamtgenauigkeit: {accuracy * 100:.2f}%")

        # Vorhersagen und Labels sammeln
        y_true = []
        y_pred = []
        for images, labels in test_dataset:
            predictions = model.predict(images, verbose=0)
            predicted_labels = np.argmax(predictions, axis=-1)
            y_true.append(labels.numpy())
            y_pred.append(predicted_labels)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Per Kategorie Auswertung
        category_counts = {name: 0 for name in class_names}
        category_correct = {name: 0 for name in class_names}

        for true_label, pred_label in zip(y_true, y_pred):
            category_name = class_names[true_label]
            category_counts[category_name] += 1
            if true_label == pred_label:
                category_correct[category_name] += 1

        print("\nGenauigkeit je Kategorie:")
        for category_name in class_names:
            count = category_counts[category_name]
            correct = category_correct[category_name]
            if count == 0:
                print(f"  {category_name}: keine Testdaten")
            else:
                print(f"  {category_name}: {correct}/{count} = {correct / count * 100:.2f}%")
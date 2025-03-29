import csv
import sys

def check_accuracy(csv_file_path):
    correct_count = 0
    total_count = 0
    truth_one_count = 0

    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            prediction = float(row[0].strip('[]'))
            truth = float(row[1].strip('[]'))
            total_count += 1

            if abs(prediction - truth) <= 0:
                correct_count += 1

            if truth == 1:
                truth_one_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Number of correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Number of 1s in truth: {truth_one_count}, about {truth_one_count / total_count * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_accuracy.py <csv_file_path>")
    else:
        check_accuracy(sys.argv[1])
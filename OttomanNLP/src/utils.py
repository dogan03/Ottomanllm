import re

def extract_loss(file_path):
    """Extract loss values and their corresponding index numbers from a flair log file."""
    loss_pattern = r'loss (\d+\.\d+)'
    accuracy_pattern = r'Accuracy (\d+\.\d+)'
    accuracy = None

    index_numbers = []
    losses = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        line_number = 0

        for line in lines:
            line_number += 1

            loss_match = re.search(loss_pattern, line)
            accuracy_match = re.search(accuracy_pattern, line)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))

            if loss_match:
                loss = float(loss_match.group(1))
                index_numbers.append(line_number)
                losses.append(loss)

    index_numbers = [i + 1 for i in range(len(index_numbers))]

    return index_numbers, losses, accuracy
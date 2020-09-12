from pathlib import Path

train_data_path = Path('./data/')
test_data_path = Path('./test/')

with open('train_annotation.txt', 'w') as file:
    for img_path in train_data_path.glob("*.jpeg"):
        name = img_path.name[:-4]
        true_label = name.split('_')[0]  # [string_label]_[idx].jpeg
        true_label = true_label.strip()
        file.write(str(img_path) + '\t' + true_label + '\n')

with open('test_annotation.txt', 'w') as file:
    for img_path in test_data_path.glob("*.jpeg"):
        name = img_path.name[:-4]
        true_label = name.split('_')[0]  # [string_label]_[idx].jpeg
        true_label = true_label.strip()
        file.write(str(img_path) + '\t' + true_label + '\n')

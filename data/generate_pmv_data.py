import sys
import csv
import pickle
import os

admnote_folder = sys.argv[1]

note_texts = {}
for file in os.listdir(admnote_folder):
    if 'csv' not in file:
        continue
    print(file)
    reader = csv.reader(open(os.path.join(admnote_folder, file)))
    next(reader, None)
    for row in reader:
        note_texts[int(row[0])] = row[1]


pmv_labels = pickle.load(open('/scratch/nkw3mr/sdoh_clinical_outcome_prediction/BEEP/Strucure-Awared-Clinical-Note-Processing/data/pmv_labels.pkl', 'rb'))
if not os.path.isdir('/scratch/nkw3mr/sdoh_clinical_outcome_prediction/clinical-outcome-prediction/tasks/mimic_iii_data/mechanical_ventilation'):
    os.mkdir('/scratch/nkw3mr/sdoh_clinical_outcome_prediction/clinical-outcome-prediction/tasks/mimic_iii_data/mechanical_ventilation')
train_file = open('/scratch/nkw3mr/sdoh_clinical_outcome_prediction/clinical-outcome-prediction/tasks/mimic_iii_data/mechanical_ventilation/pmv_train.csv', 'w')
dev_file = open('/scratch/nkw3mr/sdoh_clinical_outcome_prediction/clinical-outcome-prediction/tasks/mimic_iii_data/mechanical_ventilation/pmv_dev.csv', 'w')
test_file = open('/scratch/nkw3mr/sdoh_clinical_outcome_prediction/clinical-outcome-prediction/tasks/mimic_iii_data/mechanical_ventilation/pmv_test.csv', 'w')
train_writer = csv.writer(train_file)
dev_writer = csv.writer(dev_file)
test_writer = csv.writer(test_file)
train_writer.writerow(['id', 'text', 'label'])
dev_writer.writerow(['id', 'text', 'label'])
test_writer.writerow(['id', 'text', 'label'])

# print(pmv_labels)
print(type(pmv_labels))
print(type(note_texts))
print(f"Raw PMV labels Number: {len(pmv_labels.keys())}")
print(f"Raw Note Texts Number: {len(note_texts.keys())}")


pmv_labels_filtered = {key: value for key, value in pmv_labels.items() if key in note_texts}

print(f"Filtered PMV labels Number: {len(pmv_labels_filtered)}")
pmv_labels = pmv_labels_filtered
# print(f"Random Sample: {note_texts[199756]}")

for note in pmv_labels:
    if pmv_labels[note][-1] == 'train':
        # print(pmv_labels[note])
        train_writer.writerow([note, note_texts[note], pmv_labels[note][0]])
    if pmv_labels[note][-1] == 'val':
        dev_writer.writerow([note, note_texts[note], pmv_labels[note][0]])
    if pmv_labels[note][-1] == 'test':
        test_writer.writerow([note, note_texts[note], pmv_labels[note][0]])

train_file.close()
dev_file.close()
test_file.close()

import os
import json
import copy
import csv


def get_path(base):
    paths = []
    with os.scandir(base) as entries:
        for entry in entries:
            paths.append(base + '/' + entry.name)
    return paths

def check_dir(path):
    isExit = os.path.exists(path)
    if not isExit:
        os.mkdir(path)
    return isExit

def trimming_module_eval(target_paths, output_paths, show=True):
    def show_result(results, accuracy, precision, recall, f1):
        print('='*30)
        print('Correctly Trimmed: ', results[0])
        print('First half: ', results[1])
        print('Second half: ', results[2])
        print('Correctly predicted: ', results[3])
        print('='*30)
        print('Extra Trimmed: ', results[4])
        print('Missed: ', results[5])
        print('='*30)
        print('Total Trimmed: ', results[6])
        print('Total True: ', results[7])
        print('='*30)
        print('Accuracy: ', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1-score: ', f1)
        print('='*30)
    def search_target(target_paths, output_path):
        o_name = output_path.split('/')[-1].split('.json')[0]
        for t_path in target_paths:
            t_name = t_path.split('/')[-1]
            if o_name in t_name:
                return t_path
        return False
    def cal_confusion_matrix(correctly_predicted, extra_trimmed, missed):
        accuracy = round(correctly_predicted / (correctly_predicted + missed), 4)
        precision = round(correctly_predicted / (correctly_predicted + extra_trimmed), 4)
        recall = round(correctly_predicted / (correctly_predicted + missed), 4)
        f1 = round(2 / ((1 / recall) + (1 / precision)), 4)
        return accuracy, precision, recall, f1
    
    gotcha = 0
    first_half = 0
    second_half = 0
    extra_trimmed = 0
    missed = 0
    total_true_rallies = 0
    total_trimmed_rallies = 0
    for i in range(len(output_paths)):
        true_rallies = []
        target_path = search_target(target_paths, output_paths[i])
        if target_path:
            with open(f'{target_path}/RallySeg.csv', newline='') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    true_rallies.append(row[3:5])
                true_rallies = true_rallies[1:]
            total_true_rallies += len(true_rallies)
        else:
            continue
        
        with open(output_paths[i]) as file:
            trimmed_rallies = json.load(file)
        trimmed_rallies = trimmed_rallies['rally']
        total_trimmed_rallies += len(trimmed_rallies)

        copied_rallies = copy.deepcopy(trimmed_rallies)

        for trimmed_rally in trimmed_rallies:
            trimmed_start = int(trimmed_rally[0])
            trimmed_end = int(trimmed_rally[1])
            for i in range(len(true_rallies)):
                true_rally = true_rallies[i]
                true_start = int(true_rally[0])
                true_end = int(true_rally[1])

                if trimmed_start - 3 < true_start < true_end < trimmed_end + 3:
                    gotcha += 1
                    _ = true_rallies.pop(i)
                    copied_rallies.remove(trimmed_rally)
                    break

                if true_start < trimmed_start < true_end < trimmed_end:
                    second_half += 1
                    _ = true_rallies.pop(i)
                    copied_rallies.remove(trimmed_rally)
                    break

                if trimmed_start < true_start < trimmed_end < true_end:
                    first_half += 1
                    _ = true_rallies.pop(i)
                    copied_rallies.remove(trimmed_rally)
                    break
        extra_trimmed += len(copied_rallies)
        missed += len(true_rallies)
    
    correctly_predicted = gotcha + first_half + second_half
    results = [gotcha, first_half, second_half, correctly_predicted, extra_trimmed, missed, total_trimmed_rallies, total_true_rallies]
    accuracy, precision, recall, f1 = cal_confusion_matrix(results[3], results[4], results[5])
    if show:
        show_result(results, accuracy, precision, recall, f1)
    
    return True




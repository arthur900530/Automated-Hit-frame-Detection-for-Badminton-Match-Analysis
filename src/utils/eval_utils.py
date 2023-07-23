import torch
import os, csv, json, copy
import numpy as np
from torchvision import datasets, transforms
from models.sacnn import SACNNContainer
from models.transformer import OptimusPrimeContainer
from utils.utils import get_path, search_target

def SA_CNN_eval(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((216,384)),
            transforms.CenterCrop((216,216)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((216,384)),
            transforms.CenterCrop((216,216)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = {
        'sacnn_path': './models/weights/sacnn.pt'
    }
    sacnn = SACNNContainer(args)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = sacnn.predict(inputs, processed=True)
        if output == labels.item() == 1:
            tp += 1
        if output == 1 and labels.item() == 0:
            fp += 1
        if output == labels.item() == 0:
            tn += 1
        if output == 0 and labels.item() == 1:
            fn += 1

    accuracy, precision, recall, f1 = cal_confusion_matrix(tp, fp, fn, tn)
    print('==============================')
    print(f'SA-CNN Evaluation:')
    print('------------------')
    print(f'Training Data Count: {len(dataloaders["train"])}')
    print(f'Testing Data Count: {len(dataloaders["val"])}')
    print('------------------')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('==============================')


def cal_confusion_matrix(tp, fp, fn, tn=None):
    tn = tn if tn else 0
    accuracy = round((tp + tn) / (tp + fn + fp + tn), 4)
    precision = round(tp / (tp + fp), 4)
    recall = round(tp / (tp + fn), 4)
    f1 = round(2 / ((1 / recall) + (1 / precision)), 4)
    return accuracy, precision, recall, f1


def rally_wise_video_trimming_eval(trim_output_dir, eval_target_dir):
    gotcha = 0
    first_half = 0
    second_half = 0
    extra_trimmed = 0
    missed = 0
    total_true_rallies = 0
    total_trimmed_rallies = 0
    correctly_trimmed_ranges = {}

    trim_output_paths = get_path(trim_output_dir)
    eval_target_paths = get_path(eval_target_dir)

    for i in range(len(trim_output_paths)):
        true_rallies = []
        correctly_trimmed_list = []
        output_name = trim_output_paths[i].split('/')[-1].split('.json')[0]
    
        target_path = search_target(eval_target_paths, output_name)
        if target_path:
            with open(f'{target_path}/RallySeg.csv', newline='') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    true_rallies.append(row[3:5])
                true_rallies = true_rallies[1:]
            total_true_rallies += len(true_rallies)
        else:
            continue
        
        with open(trim_output_paths[i]) as file:
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
                    correctly_trimmed_list.append((true_start, true_end))
                    gotcha += 1
                    _ = true_rallies.pop(i)
                    copied_rallies.remove(trimmed_rally)
                    break

                elif true_start < trimmed_start < true_end < trimmed_end:
                    correctly_trimmed_list.append((trimmed_start, true_end))
                    second_half += 1
                    _ = true_rallies.pop(i)
                    copied_rallies.remove(trimmed_rally)
                    break

                elif trimmed_start < true_start < trimmed_end < true_end:
                    correctly_trimmed_list.append((true_start, trimmed_end))
                    first_half += 1
                    _ = true_rallies.pop(i)
                    copied_rallies.remove(trimmed_rally)
                    break
            correctly_trimmed_ranges[output_name] = correctly_trimmed_list
            
        extra_trimmed += len(copied_rallies)
        missed += len(true_rallies)
    
    correctly_predicted = gotcha + first_half + second_half
    accuracy, precision, recall, f1 = cal_confusion_matrix(correctly_predicted, extra_trimmed, missed)
    print('==============================')
    print('Correctly predicted: ', correctly_predicted)
    print('Extra Trimmed: ', extra_trimmed)
    print('Missed: ', missed)
    print('------------------')
    print('Total Trimmed: ', total_trimmed_rallies)
    print('Total True: ', total_true_rallies)
    print('------------------')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('==============================')
    
    return correctly_trimmed_ranges


def transformer_eval(data_dir):
    def top_bottom(joint):
        a = joint[0][15][1] + joint[0][16][1]
        b = joint[1][15][1] + joint[1][16][1]
        if a > b:
            top = 1
            bottom = 0
        else:
            top = 0
            bottom = 1
        return top, bottom
    
    def get_val_data(root, sc=None):  
        c_count = 0
        n_count = 0
        
        data_x = []
        data_y = []
        
        folder_paths = get_path(root)
        
        for path in folder_paths:
            vid_name = path.split('/')[-1]
            print(vid_name)
            
            score_json_paths = get_path(path)
            for json_path in score_json_paths:
            
                with open(json_path, 'r') as score_json:
                    frame_dict = json.load(score_json)
                    
                score_x = []
                score_y = []
            
                for i in range(len(frame_dict['frames'])):
                    label = frame_dict['frames'][i]['label']
                    score_y.append(frame_dict['frames'][i]['label'])
                    joint = np.array(frame_dict['frames'][i]['joint'])
                            
                    top, bot = top_bottom(joint)
                    if top != 1:
                        c_count += 1
                        t = []
                        t.append(joint[bot])
                        t.append(joint[top])
                        joint = np.array(t)
                    else:
                        n_count += 1
                        
                    joint = np.array(joint)                # 2, 17, 2
                    joint = np.reshape(joint, [1,-1])
                    joint = sc.transform(joint)
                    joint = np.reshape(joint, [2,17,2])
        
                    score_x.append(joint)
                    
                score_x = np.array(score_x)
                score_y = np.array(score_y)
                score_y = np.reshape(score_y, [score_y.shape[0],])
                
                data_x.append(score_x)
                data_y.append(score_y)
        return data_x, data_y
    
    print('==============================')
    print('Test Matches:')
    
    args = {
        'opt_path': './models/weights/opt.pt',
        'scaler_path': './models/weights/scaler.pickle'
    }
    opt = OptimusPrimeContainer(args)
    data_x_val, data_y_val = get_val_data(data_dir, opt.scaler)
    data_count = 0
    for i in range(len(data_x_val)):
        data_count += len(data_x_val[i])

    d_zero = {
    'tp': 0,
    'fp': 0,
    'fn': 0,
    'tn': 0
    }
    d_one = {
    'tp': 0,
    'fp': 0,
    'fn': 0,
    'tn': 0
    }
    d_two = {
    'tp': 0,
    'fp': 0,
    'fn': 0,
    'tn': 0
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(len(data_x_val)):
        input_seq = torch.tensor(np.array(data_x_val[i])).unsqueeze(0).to(torch.float32).to(device)
        labels = data_y_val[i]
        result = opt.predict(input_sequence=input_seq, scaled=True)
        result = result.cpu().numpy()

        for j in range(len(labels)):
            predicted_d = result[j]
            target_d = labels[j]

            if predicted_d == target_d == 0:
                d_zero['tp'] += 1
            elif predicted_d != target_d and predicted_d == 0:
                d_zero['fp'] += 1
            elif predicted_d == target_d and target_d != 0:
                d_zero['tn'] += 1
            elif predicted_d != target_d and target_d == 0:
                d_zero['fn'] += 1

            if predicted_d == target_d == 1:
                d_one['tp'] += 1
            elif predicted_d != target_d and predicted_d == 1:
                d_one['fp'] += 1
            elif predicted_d == target_d and target_d != 1:
                d_one['tn'] += 1
            elif predicted_d != target_d and target_d == 1:
                d_one['fn'] += 1

            if predicted_d == target_d == 2:
                d_two['tp'] += 1
            elif predicted_d != target_d and predicted_d == 2:
                d_two['fp'] += 1
            elif predicted_d == target_d and target_d != 2:
                d_two['tn'] += 1
            elif predicted_d != target_d and target_d == 2:
                d_two['fn'] += 1

    accuracy_0, precision_0, recall_0, f1_0 = cal_confusion_matrix(d_zero['tp'], d_zero['fp'], d_zero['fn'], d_zero['tn'])
    accuracy_1, precision_1, recall_1, f1_1 = cal_confusion_matrix(d_one['tp'], d_one['fp'], d_one['fn'], d_one['tn'])
    accuracy_2, precision_2, recall_2, f1_2 = cal_confusion_matrix(d_two['tp'], d_two['fp'], d_two['fn'], d_two['tn'])

    print('==============================')
    print(f'Transformer Evaluation:')
    print(f'Testing Data Keypoint Sequences: {len(data_x_val)}')
    print(f'Testing Data Keypoint Pairs: {data_count}')
    print('------------------')
    print('d: 0')
    print(f'Accuracy: {accuracy_0:.4f}')
    print(f'Precision: {precision_0:.4f}')
    print(f'Recall: {recall_0:.4f}')
    print(f'F1-score: {f1_0:.4f}')
    print('------------------')
    print('d: 1')
    print(f'Accuracy: {accuracy_1:.4f}')
    print(f'Precision: {precision_1:.4f}')
    print(f'Recall: {recall_1:.4f}')
    print(f'F1-score: {f1_1:.4f}')
    print('------------------')
    print('d: 2')
    print(f'Accuracy: {accuracy_2:.4f}')
    print(f'Precision: {precision_2:.4f}')
    print(f'Recall: {recall_2:.4f}')
    print(f'F1-score: {f1_2:.4f}')
    print('==============================')


def hit_frame_detection_eval(data_dir, data_dir_2, output_dir, intervals, correctly_trimmed_ranges):
    def top_bottom(joint):
        a = joint[0][15][1] + joint[0][16][1]
        b = joint[1][15][1] + joint[1][16][1]
        if a > b:
            top = 1
            bottom = 0
        else:
            top = 0
            bottom = 1
        return top, bottom
    
    def get_val_data(root, sc=None):  
        c_count = 0
        n_count = 0
        
        data_x = []
        data_y = []
        
        folder_paths = get_path(root)
        
        for path in folder_paths:
            vid_name = path.split('/')[-1]
            print(vid_name)
            
            score_json_paths = get_path(path)
            for json_path in score_json_paths:
            
                with open(json_path, 'r') as score_json:
                    frame_dict = json.load(score_json)
                    
                score_x = []
                score_y = []
            
                for i in range(len(frame_dict['frames'])):
                    label = frame_dict['frames'][i]['label']
                    score_y.append(frame_dict['frames'][i]['label'])
                    joint = np.array(frame_dict['frames'][i]['joint'])
                            
                    top, bot = top_bottom(joint)
                    if top != 1:
                        c_count += 1
                        t = []
                        t.append(joint[bot])
                        t.append(joint[top])
                        joint = np.array(t)
                    else:
                        n_count += 1
                        
                    joint = np.array(joint)                # 2, 17, 2
                    joint = np.reshape(joint, [1,-1])
                    joint = sc.transform(joint)
                    joint = np.reshape(joint, [2,17,2])
        
                    score_x.append(joint)
                    
                score_x = np.array(score_x)
                score_y = np.array(score_y)
                score_y = np.reshape(score_y, [score_y.shape[0],])
                
                data_x.append(score_x)
                data_y.append(score_y)
        return data_x, data_y

    def check_hit_frame(direction_list):
        '''
        0 -> 1
        0 -> 2
        1 -> 2
        2 -> 1
        '''
        last_direction = 0
        hit_frame_indices = []
        for i in range(len(direction_list)):
            direction = direction_list[i]
            if direction != last_direction:
                if last_direction == 0:
                    hit_frame_indices.append(i)
                elif last_direction == 1:
                    if direction == 2:
                        hit_frame_indices.append(i)
                    else:
                        continue
                elif last_direction == 2:
                    if direction == 1:
                        hit_frame_indices.append(i)
                    else:
                        continue
            last_direction = direction
        hit_frame_indices = np.array(hit_frame_indices).astype(float)

        return hit_frame_indices
    
    def check_range(ranges, frame_num):
        for r in ranges:
            if r[0] < frame_num < r[1]:
                return True, r[0], r[1]
        return False, r[0], r[1]
    print('==============================')
    print('Test Matches:')
    args = {
            'opt_path': './models/weights/opt.pt',
            'scaler_path': './models/weights/scaler.pickle'
        }
    opt = OptimusPrimeContainer(args)
    data_x_val, data_y_val = get_val_data(data_dir, opt.scaler)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_paths = get_path(output_dir)
    eval_target_paths = get_path(data_dir_2)

    for j in range(len(intervals)):
        interval = intervals[j]
        gotcha = 0
        extra_pridected = 0
        missed = 0
        tn = 0

        for i in range(len(data_x_val)):
            input_seq = torch.tensor(np.array(data_x_val[i])).unsqueeze(0).to(torch.float32).to(device)
            labels = data_y_val[i]
            result = opt.predict(input_sequence=input_seq, scaled=True)
            result = result.cpu().numpy()

            true_hit_frames = list(check_hit_frame(data_y_val[i]).astype(int))
            predicted_hit_frames = check_hit_frame(result).astype(int)
            copied_predict = copy.deepcopy(predicted_hit_frames)
            got = 0
            for hit_frame in predicted_hit_frames:
                for i in range(len(true_hit_frames)):
                    true_hit_frame = int(true_hit_frames[i])
                    if true_hit_frame - interval <= hit_frame <= true_hit_frame + interval:
                        gotcha += 1
                        got += 1
                        _ = true_hit_frames.pop(i)
                        copied_predict = np.delete(copied_predict, np.where(copied_predict == hit_frame))
                        break

            extra_pridected += len(copied_predict)
            missed += len(true_hit_frames)
            tn = tn + len(labels) - len(copied_predict) - len(true_hit_frames) - got

        gotcha_2 = 0
        extra_pridected_2 = 0
        missed_2 = 0
        fn_list = []

        for video_path in video_paths:
            video_name = video_path.split('/')[-1]
            target_eval_path = search_target(eval_target_paths, video_name)

            if target_eval_path:
                csv_paths = get_path(f'{target_eval_path}/label')
                if j == 0:
                    print(video_name)
            else:
                continue
            
            ranges = correctly_trimmed_ranges[video_name]
            output_paths = get_path(f'{video_path}')

            # true hit frame
            true_hit_frames = []
            for csv_path in csv_paths:
                with open(csv_path, newline='') as csvfile:
                    rows = csv.reader(csvfile)
                    for row in rows:
                        if row[3] != 'frame_num':
                            true_hit_frames.append(row[3])
            
            for true_hit_frame in true_hit_frames:
                valid, r1, r2 = check_range(ranges, int(true_hit_frame))
                if not valid:
                    true_hit_frames.remove(true_hit_frame)
                else:
                    k = str(r1) + '_' + str(r2)
                    if not k in fn_list:
                        fn_list.append(k)
            
            # predicted hit frames
            predicted_hit_frames = np.array([])
            for output_path in output_paths:
                with open(output_path) as file:
                    joints_info = json.load(file)
                    hit_frames = joints_info["hit frames"]
                predicted_hit_frames = np.concatenate((predicted_hit_frames, hit_frames), axis=None)
            predicted_hit_frames = list(predicted_hit_frames)
            
            for predicted_hit_frame in predicted_hit_frames:
                valid, r1, r2 = check_range(ranges, int(predicted_hit_frame))
                if not valid:
                    predicted_hit_frames.remove(predicted_hit_frame)
                else:
                    k = str(r1) + '_' + str(r2)
                    if not k in fn_list:
                        fn_list.append(k)

            copied_predict = copy.deepcopy(predicted_hit_frames)
            
            for hit_frame in predicted_hit_frames:
                for i in range(len(true_hit_frames)):
                    true_hit_frame = int(true_hit_frames[i])
                    if true_hit_frame - interval <= hit_frame <= true_hit_frame + interval:
                        gotcha_2 += 1
                        _ = true_hit_frames.pop(i)
                        copied_predict = np.delete(copied_predict, np.where(copied_predict == hit_frame))
                        break

            extra_pridected_2 += len(copied_predict)
            missed_2 += len(true_hit_frames)

        total_frames = 0
        for ranges in fn_list:
            r1 = int(ranges.split('_')[0])
            r2 = int(ranges.split('_')[1])
            total_frames = total_frames + (r2 - r1)
        tn_2 = total_frames - gotcha_2 - extra_pridected_2 - missed_2

        total_tp = gotcha + gotcha_2
        total_fp = extra_pridected + extra_pridected_2
        total_fn = missed + missed_2
        total_tn = tn + tn_2

        accuracy, precision, recall, f1 = cal_confusion_matrix(total_tp, total_fp, total_fn, total_tn)
        if j == 0:
            print('==============================')
        print(f'Interval: +-{interval}')
        print('------------------')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print('==============================')
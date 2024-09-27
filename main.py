import datetime
import random
import numpy as np
from tqdm import tqdm
import torch
import pickle
import time
import os
import math
import pandas as pd
from collections import Counter

import settings
from model import model
from results.data_reader import print_output_to_file, calculate_average, clear_log_meta_model

device = settings.gpuId if torch.cuda.is_available() else 'cpu'
city = settings.city

id2pop_dict = pickle.load(open(f'./processed_data/{city}_id2pop_dict', 'rb'))
id2pop_group = pickle.load(open(f'./processed_data/{city}_id2PopGroup_dict', 'rb'))
pop_group = max(list(id2pop_group.values()))+1
dist_matrix = pd.read_csv(f'./processed_data/{city}_distance_matrix.csv')
hourPopCat = pickle.load(open(f'./processed_data/{city}_hourPopCat_dict', 'rb'))
poi2cat = pickle.load(open(f'./processed_data/{city}_poi2cat_dict', 'rb'))
poi2cat = dict(sorted(poi2cat.items()))

def temporal_dist_sample(dist_filter, poi2cat, cat_dist, num_samples):
    sampled_pois_pos, sampled_pois_neg = [],[]
    available_pois = list(dist_filter.keys())
    num_samples = min(num_samples, int(len(available_pois)/2))
    for _ in range(num_samples):
        sampled_poi = random.choices(
            population=available_pois,
            weights=[cat_dist.get(poi2cat[poi],0) for poi in available_pois],
            k=1
        )[0]
        sampled_pois_pos.append(sampled_poi)
        available_pois.remove(sampled_poi)
        sampled_poi = random.choices(
            population=available_pois,
            weights=[1 - cat_dist.get(poi2cat[poi],0) for poi in available_pois],
            k=1
        )[0]
        sampled_pois_neg.append(sampled_poi)
        available_pois.remove(sampled_poi)
    return sampled_pois_pos, sampled_pois_neg

def generate_sample_to_device(sample):
    sample_to_device = []
    for seq in sample:
        features = torch.tensor(seq[:7]).to(device) # l,c,u,t,d,pop,dist
        day_nums = torch.tensor(seq[7]).to(device)
        sample_to_device.append((features, day_nums))

    return sample_to_device

def CL_pos_neg_sampling(sample, current_POI, short_seq, short_dist_seq, short_hour_seq, short_cat_seq, time_window, dist_window):
    pos_sample_to_device_list, neg_sample_to_device_list = [], []
    long_term_sequences = sample[:-1]
    poi_seq, cat_seq, hour_seq, dist_seq =[],[],[],[]
    past_cat_list, past_dist_list = [],[]
    for seq in long_term_sequences:
        poi_seq.extend(seq[0])
        cat_seq.extend(seq[1])
        hour_seq.extend(seq[3])
        dist_seq.extend(seq[6])
    poi_seq.extend(short_seq[:-1])
    cat_seq.extend(short_cat_seq[:-1])
    hour_seq.extend(short_hour_seq[:-1])
    dist_seq.extend(sample[-1][6][:-1])
    # past cat checked within [gt_hour-time_window, gt_hour +time_window]
    gt_hour = short_hour_seq[-1]
    for i in range(len(hour_seq)):
        if (hour_seq[i] == gt_hour) | (hour_seq[i] == (gt_hour-time_window)%24) | (hour_seq[i] == (gt_hour+time_window)%24):
            past_cat_list.append(cat_seq[i])
            past_dist_list.append(dist_seq[i])
    # the ration each category checked in the past gt hour period.
    counter_cat = Counter(past_cat_list)
    cat_distribute = {key: value / len(past_cat_list) for key, value in counter_cat.items()}
    counter_dist = Counter(past_dist_list)
    dist_distribute = {key: value / len(past_dist_list) for key, value in counter_dist.items()}
    # print(cat_distribute, dist_distribute)
    if settings.sample_feature == 'dist':
        # avg dist of short seq
        avg_dist = sum(short_dist_seq[:-1])/(len(short_dist_seq)-2)
        cur_cand_dist = (dist_matrix.loc[current_POI]-avg_dist).to_dict()# distanc between cand and lt-1 close to the average short seq dist
        # cur_cand_dist = (dist_matrix.loc[current_POI]).to_dict()# distanc between cand and lt-1 close to 0
        sorted_list = sorted(cur_cand_dist.items(), key=lambda x: abs(x[1]), reverse=False)
        pos_sample = [int(x[0]) for x in sorted_list[:settings.cl_sample-1]] + [current_POI]
        neg_sample = [int(x[0]) for x in sorted_list if int(x[0]) not in short_seq][-settings.cl_sample:]
    if settings.sample_feature == 'cat':
        pos_sample,neg_sample = temporal_dist_sample(poi2cat, poi2cat, cat_distribute, settings.cl_sample)
    if settings.sample_feature == 'all':   
        if len(list(dist_distribute.keys())) == 0:
            sampled_dist = 5
        else:
            sampled_dist = random.choices( population = list(dist_distribute.keys()), weights = list(dist_distribute.values()), k=1)[0]
        cur_cand_dist = (dist_matrix.loc[current_POI]).to_dict()
        dist_filter = {int(key): value for key, value in cur_cand_dist.items() if int(value) <= sampled_dist+dist_window} # dist with lt-1 <=5km
        pos_sample,neg_sample = temporal_dist_sample(dist_filter, poi2cat, cat_distribute, settings.cl_sample)
        
    pos_dist = [int(dist_matrix.loc[current_POI][x]) for x in pos_sample]
    neg_dist = [int(dist_matrix.loc[current_POI][x]) for x in neg_sample]
    pos_cat = [poi2cat[x] for x in pos_sample]
    neg_cat = [poi2cat[x] for x in neg_sample]
    pos_sample_to_device_list.append(pos_sample)
    pos_sample_to_device_list.append(pos_cat)
    pos_sample_to_device_list.append(pos_dist)
    neg_sample_to_device_list.append(neg_sample)
    neg_sample_to_device_list.append(neg_cat)
    neg_sample_to_device_list.append(neg_dist)

    return torch.tensor(pos_sample_to_device_list).to(device), torch.tensor(neg_sample_to_device_list).to(device)



def train_model(train_set, test_set, h_params, vocab_size, device, run_name):
    torch.cuda.empty_cache()
    model_path = f"./results/{run_name}_model"
    log_path = f"./results/{run_name}_log"
    meta_path = f"./results/{run_name}_meta"

    print("parameters:", h_params)

    if os.path.isfile(f'./results/{run_name}_model'):
        try:
            os.remove(f"./results/{run_name}_meta")
            os.remove(f"./results/{run_name}_model")
            os.remove(f"./results/{run_name}_log")
        except OSError:
            pass
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()

    # construct model
    rec_model = model(
        vocab_size=vocab_size,
        f_embed_size=h_params['embed_size'],
        num_encoder_layers=h_params['tfp_layer_num'],
        num_lstm_layers=h_params['lstm_layer_num'],
        num_heads=h_params['head_num'],
        forward_expansion=h_params['expansion'],
        dropout_p=h_params['dropout']
    )

    rec_model = rec_model.to(device)

    # Continue with previous training
    start_epoch = 0
    if os.path.isfile(model_path):
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()

        meta_file = open(meta_path, "rb")
        start_epoch = pickle.load(meta_file) + 1
        meta_file.close()

    params = list(rec_model.parameters())

    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    loss_dict, recalls, ndcgs, maps, fdcgs, ddcgs = {}, {}, {}, {}, {}, {}

    for epoch in range(start_epoch, h_params['epoch']):
        print(f'-------------------------epoch: {epoch}--------------------------------')
        begin_time = time.time()
        total_loss = 0.
        for sample in tqdm(train_set):
            sample_to_device = generate_sample_to_device(sample)
            pos_sample_to_device_list = []
            neg_sample_to_device_list = []
            # current_POI = sample[-1][0][-1] # gt
            current_POI = sample[-1][0][-2] # last loc
            user_id = sample[-1][2][0]
            short_seq = sample[-1][0]
            short_dist_seq = sample[-1][-2]
            short_cat_seq = sample[-1][1]
            short_hour_seq = sample[-1][3]

            # add the distance interval info between the last loc and each candidate
            cand = []
            # cand.append(list(id2pop_dict.keys()))
            # cand.append(list(id2pop_dict.values()))
            cand.append(list(poi2cat.keys()))
            cand.append(list(poi2cat.values()))


            dist_last_cand = list(dist_matrix.loc[current_POI])
            dist_last_cand = [int(x) for x in dist_last_cand]
            cand.append(dist_last_cand)
            cand = torch.tensor(cand).to(device)

            # CL sample
            pos_sample_to_device_list, neg_sample_to_device_list = CL_pos_neg_sampling(sample, current_POI, short_seq, short_dist_seq, short_cat_seq, short_hour_seq, h_params['time_window'], h_params['dist_window'])
            loss, _ = rec_model(sample_to_device, pos_sample_to_device_list, neg_sample_to_device_list, cand)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        print(f'-------------------test for epoch {epoch} on test set--------------------------')
        recall, ndcg, map, fdcg, ddcg = test_model(test_set, rec_model, epoch)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map
        fdcgs[epoch] = fdcg
        ddcgs[epoch] = ddcg

        # Record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[epoch] = avg_loss
        print(f"epoch: {epoch}; average loss: {avg_loss}, time taken: {int(time.time() - begin_time)}s")
        # Save model
        torch.save(rec_model.state_dict(), model_path)
        # Save last epoch
        meta_file = open(meta_path, 'wb')
        pickle.dump(epoch, meta_file)
        meta_file.close()

        # Early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss) > 10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {epoch}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        pickle.dump(fdcgs, file)
        pickle.dump(ddcgs, file)
        file.close()

    print("============================")


def test_model(test_set, rec_model, epoch, ks=[1, 5, 10]):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]
    
    # def calc_fdcg(labels, preds, pop_dict, k): # pop version
    #     fdcg, samples = 0.0, 0
    #     pre_pop = np.array([[pop_dict.get(element) for element in row] for row in preds[:,:k].numpy()], dtype=float)
    #     label_pop = np.array([pop_dict.get(element[0]) for element in labels.numpy()], dtype=float)
    #     label_pop = label_pop[:, np.newaxis]
    #     pre_pop_diff = pre_pop - label_pop
    #     for row in pre_pop_diff:
    #         for v in range(k):
    #             if abs(row[v]) >= 0:
    #                 fdcg += abs(row[v]) / math.log2(v + 1 + 1)
    #                 samples+=1
    #     if samples==0:
    #         return 'nan'
    #     else:
    #         return torch.tensor(fdcg/samples)
    def calc_fdcg(labels, preds, poi2cat, k): # temporal version
        fdcg = 0.0
        pre_cat = np.array([[poi2cat.get(element) for element in row] for row in preds[:,:k].numpy()])
        label_cat = np.array([poi2cat.get(element[0]) for element in labels.numpy()])
        for i in range(len(pre_cat)):
            for v in range(k):
                if pre_cat[i][v] == label_cat[i]:
                    fdcg += 1 / math.log2(v + 1 + 1)
        return torch.tensor(fdcg/labels.shape[0])

    def calc_ddcg(labels, preds, dist_matrix, k):
        # pred dist with gt close to 0
        ddcg, samples = 0.0, 0
        pre_dist = []
        for i in range(len(labels)):
            gt = labels.numpy()[i][0]
            pre_dist_gt = [dist_matrix.loc[gt][element] for element in preds[i,:k].numpy()]
            pre_dist.append(pre_dist_gt)
        pre_pop_diff = np.array(pre_dist, dtype=float)
        for row in pre_pop_diff:
            for v in range(k):
                if abs(row[v]) >= 0:
                    ddcg += abs(row[v]) / math.log2(v + 1 + 1)
                    samples+=1
        if samples==0:
            return 'nan'
        else:
            return torch.tensor(ddcg/samples)
    
    
    preds, labels = [], []
    for sample in tqdm(test_set):
        sample_to_device = generate_sample_to_device(sample)
        pos_sample_to_device_list = []
        neg_sample_to_device_list = []
        # current_POI = sample[-1][0][-1] # gt
        current_POI = sample[-1][0][-2] # last loc
        user_id = sample[-1][2][0]
        short_seq = sample[-1][0]
        short_dist_seq = sample[-1][-2]
        short_cat_seq = sample[-1][1]
        short_hour_seq = sample[-1][3]

        # add the distance interval info between the last loc and each candidate
        cand = []
        # cand.append(list(id2pop_dict.keys()))
        # cand.append(list(id2pop_dict.values()))
        cand.append(list(poi2cat.keys()))
        cand.append(list(poi2cat.values()))
        dist_last_cand = list(dist_matrix.loc[current_POI])
        dist_last_cand = [int(x) for x in dist_last_cand]
        cand.append(dist_last_cand)
        cand = torch.tensor(cand).to(device)

        pos_sample_to_device_list, neg_sample_to_device_list = CL_pos_neg_sampling(sample, current_POI, short_seq, short_dist_seq, short_cat_seq, short_hour_seq, h_params['time_window'], h_params['dist_window'])

        pred, label = rec_model.predict(sample_to_device, pos_sample_to_device_list, neg_sample_to_device_list, cand)
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0).cpu()
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1).cpu()

    recalls, NDCGs, MAPs, FDCGs, DDCGs= {}, {}, {}, {}, {}

    

    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        FDCGs[k] = calc_fdcg(labels, preds, poi2cat, k)
        DDCGs[k] = calc_ddcg(labels, preds, dist_matrix, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]},\tFDCG@{k} : {FDCGs[k]},\tDDCG@{k} : {DDCGs[k]}")

    return recalls, NDCGs, MAPs, FDCGs, DDCGs


if __name__ == '__main__':
    # Get current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Datetime of now:", now_str)

    # Get parameters
    h_params = {
        'expansion': 4,
        'random_mask': settings.enable_random_mask,
        'mask_prop': settings.mask_prop,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'loss_delta': 1e-3}

    processed_data_directory = './processed_data/'

    # Read training data
    file = open(f"{processed_data_directory}/{city}_train", 'rb')
    train_set = pickle.load(file)
    file = open(f"{processed_data_directory}/{city}_valid", 'rb')
    valid_set = pickle.load(file)

    # Read meta data
    file = open(f"{processed_data_directory}/{city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {"POI": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device),
                  "dist": torch.tensor(len(meta["dist"])).to(device),
                  "pop": torch.tensor(pop_group).to(device)}

    # Adjust specific parameters for each city
    if city == 'SIN':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 3
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
        h_params['time_window'] = 1
        h_params['dist_window'] = 2
    elif city == 'NYC':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 3
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
        h_params['time_window'] = 1
        h_params['dist_window'] = 2
    elif city == 'PHO':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 4
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
        h_params['time_window'] = 1
        h_params['dist_window'] = 3

    # Create output folder
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    print(f'Current GPU {settings.gpuId}')
    for run_num in range(1, 1 + settings.run_times):
        run_name = f'{settings.output_file_name} {run_num}'
        print(run_name)

        train_model(train_set, valid_set, h_params, vocab_size, device, run_name=run_name)
        print_output_to_file(settings.output_file_name, run_num)

        t = random.randint(1, 9)
        print(f"sleep {t} seconds")
        time.sleep(t)

        clear_log_meta_model(settings.output_file_name, run_num)
    calculate_average(settings.output_file_name, settings.run_times)

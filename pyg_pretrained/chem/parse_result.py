### Parsing the result!
import tensorflow as tf
import os
import numpy as np
import pickle

def get_test_acc(event_file):
    val_auc_list = np.zeros(100)
    test_auc_list = np.zeros(100)
    for e in list(tf.train.summary_iterator(event_file)):
        if len(e.summary.value) == 0:
            continue
        if e.summary.value[0].tag == "data/val_auc":
            val_auc_list[e.step-1] = e.summary.value[0].simple_value
        if e.summary.value[0].tag == "data/test_auc":
            test_auc_list[e.step-1] = e.summary.value[0].simple_value
    
    best_epoch = np.argmax(val_auc_list)

    return test_auc_list[best_epoch]

if __name__ == "__main__":

    dataset_list = ["muv", "bace", "bbbp", "clintox", "hiv", "sider", "tox21", "toxcast"]
    #10 random seed
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config_list = []

    config_list.append("gin_nopretrain")
    config_list.append("gin_infomax")
    config_list.append("gin_edgepred")
    config_list.append("gin_masking")
    config_list.append("gin_contextpred")
    config_list.append("gin_supervised")
    config_list.append("gin_supervised_infomax")
    config_list.append("gin_supervised_edgepred")
    config_list.append("gin_supervised_masking")
    config_list.append("gin_supervised_contextpred")
    config_list.append("gcn_nopretrain")
    config_list.append("gcn_supervised_contextpred")
    config_list.append("graphsage_nopretrain")
    config_list.append("graphsage_supervised_contextpred")
    config_list.append("gat_nopretrain")
    config_list.append("gat_supervised_contextpred")

    result_mat = np.zeros((len(seed_list), len(config_list), len(dataset_list)))

    for i, seed in enumerate(seed_list):
        for j, config in enumerate(config_list):
            for k, dataset in enumerate(dataset_list):
                dir_name = "runs/finetune_cls_runseed" + str(seed) + "/" + dataset + "/" + config
                print(dir_name)
                file_in_dir = os.listdir(dir_name)
                event_file_list = []
                for f in file_in_dir:
                    if "events" in f:
                        event_file_list.append(f)

                event_file = event_file_list[0]

                result_mat[i, j, k] = get_test_acc(dir_name + "/" + event_file)

    with open("result_summary", "wb") as f:
        pickle.dump({"result_mat": result_mat, "seed_list": seed_list, "config_list": config_list, "dataset_list": dataset_list}, f)










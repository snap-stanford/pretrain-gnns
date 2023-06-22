import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
import seaborn as sns
import copy

def analysis():
    # construct result dict
    split = 'species'
    parent_result_path = 'result'

    result_dict = {}  
    for seed_result_path in ["finetune_seed" + str(i) for i in range(10)]:
        result_dict[seed_result_path] = {}
        seed_result_names = os.listdir(os.path.join(parent_result_path, seed_result_path))
        filtered_seed_result_names = seed_result_names #[n for n in seed_result_names if split in n]
        #filtered_seed_result_names = ["speciessplit_cbow_l1-1_center0_epoch100", "speciessplit_gae_epoch100", "speciessplit_supervised_drop0.2_cbow_l1-1_center0_epoch100_epoch100"]
        for name in filtered_seed_result_names:
            #print(name)
            with open(os.path.join(parent_result_path, seed_result_path, name), "rb") as f:
                result = pickle.load(f)
                result['train'] = result['train']
                result['val'] = result['val']
                result['test_easy'] = result['test_easy']
                result['test_hard'] = result['test_hard']
            result_dict[seed_result_path][name] = result


    #top_k = 40
    best_result_dict = {}  # dict[SEED#][experiment][test_easy/test_hard] = np.array, dim top_k classes
    for seed in result_dict:
        #print(seed)
        best_result_dict[seed] = {}
        for experiment in result_dict[seed]:
            #print(experiment)
            best_result_dict[seed][experiment] = {}
            #val = result_dict[seed][experiment]["val"][:, :top_k]  # look at the top k classes
            val = result_dict[seed][experiment]["val"]
            val_ave = np.average(val, axis = 1)
            best_epoch = np.argmax(val_ave)
            
            #test_easy = result_dict[seed][experiment]["test_easy"][:, :top_k]
            test_easy = result_dict[seed][experiment]["test_easy"]
            test_easy_best = test_easy[best_epoch]
            
            #test_hard = result_dict[seed][experiment]["test_hard"][:, :top_k]
            test_hard = result_dict[seed][experiment]["test_hard"]
            test_hard_best = test_hard[best_epoch]
            
            best_result_dict[seed][experiment]["test_easy"] = test_easy_best
            best_result_dict[seed][experiment]["test_hard"] = test_hard_best

    # average across the top k tasks and then average across all the seeds
    mean_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    std_result_dict = {}  # dict[experiment][test_easy/test_hard] = float
    for experiment in filtered_seed_result_names:
        print(experiment)
        mean_result_dict[experiment] = {}
        std_result_dict[experiment] = {}
        test_easy_list = []
        test_hard_list = []
        for seed in best_result_dict:
            print(seed)
            test_easy_list.append(best_result_dict[seed][experiment]['test_easy'])
            test_hard_list.append(best_result_dict[seed][experiment]['test_hard'])
        mean_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).mean()
        mean_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).mean()
        std_result_dict[experiment]['test_easy'] = np.array(test_easy_list).mean(axis=1).std()
        std_result_dict[experiment]['test_hard'] = np.array(test_hard_list).mean(axis=1).std()
    # copy mean_result_dict and std_result_dict for printing of latex table format at end of notebook
    species_mean_result_dict = copy.deepcopy(mean_result_dict)
    species_std_result_dict = copy.deepcopy(std_result_dict)

    # results test hard
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test_hard'], reverse=True)
    for k, _ in sorted_test_hard:
        print(k)
        print('{} +- {}'.format(mean_result_dict[k]['test_hard']*100, std_result_dict[k]['test_hard']*100))
        print("")


    os.makedirs("figures", exist_ok=True)

    # plot rocs comparison graphs and save
    def plot_scatter(x_data, y_data):
        ax = sns.scatterplot(x=x_data, y=y_data)
        ax.plot([0, 1], [0, 1], 'red', linewidth=1)
        ax.set(ylim=(0, 1))
        ax.set(xlim=(0, 1))
        return ax

    def get_fig_name(method_y, method_x):
        fig_name = 'bio_pairwise_' +  method_y + '_vs_' + method_x + '.pdf'
        return fig_name

    def save_fig(method_y, method_x, ax):
        fig_name = get_fig_name(method_y, method_x)
        fig = ax.get_figure()
        fig.savefig(os.path.join('figures', fig_name))

    # first get average auc score across 10 seeds for each of the 50 tasks for test_hard
    mean_task_result_dict = {}  # dict[experiment] = np.array, dim top_k classes 
    for experiment in filtered_seed_result_names:
        # print(experiment)
        test_hard_list = []
        for seed in best_result_dict:
            # print(seed)
            test_hard_list.append(best_result_dict[seed][experiment]['test_hard'])
        mean_task_result_dict[experiment] = np.array(test_hard_list).mean(axis=0)

    experiment_pairs = [
        ('_supervised_masking', '_supervised'),
        ('_supervised_masking', '_masking'),
        ('_supervised_masking', '_nopretrain'),
        ('_masking', '_nopretrain'),
        ('_supervised', '_nopretrain'),
        ('_supervised_contextpred', '_supervised'),
        ('_supervised_contextpred', '_contextpred'),
        ('_supervised_contextpred', '_nopretrain'),
        ('_contextpred', '_nopretrain'),
    ]

    for architecture in ['gin']: 
        for exp_y, exp_x in experiment_pairs:
            method_y = architecture + exp_y
            method_x = architecture + exp_x
            y_data, x_data = mean_task_result_dict[method_y], mean_task_result_dict[method_x]
            ax = plot_scatter(x_data, y_data)
            fig = ax.get_figure()
            fig.set_size_inches(6, 6)
            save_fig(method_y, method_x, ax)
            plt.close(fig) 

            if exp_x == '_nopretrain':
                print("Negative transfer of " + exp_y[1:])
                print(np.sum(x_data > y_data + 0.001))
                
    

### to draw training curves.
def draw_training():
    pass

if __name__ == "__main__":
    analysis()

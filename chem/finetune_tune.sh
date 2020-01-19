#### GIN fine-tuning
runseed=$1
device=$2
split=scaffold

### for GIN
for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
do
for unsup in contextpred infomax edgepred masking
do
model_file=${unsup}
python finetune.py --input_model_file model_gin/${model_file}.pth --split $split --filename ${dataset}/gin_${model_file} --device $device --runseed $runseed --gnn_type gin --dataset $dataset

model_file=supervised_${unsup}
python finetune.py --input_model_file model_gin/${model_file}.pth --split $split --filename ${dataset}/gin_${model_file} --device $device --runseed $runseed --gnn_type gin --dataset $dataset
done

python finetune.py --split $split --filename ${dataset}/gin_nopretrain --device $device --runseed $runseed --gnn_type gin --dataset $dataset
python finetune.py --split $split --input_model_file model_gin/supervised.pth --filename ${dataset}/gin_supervised --device $device --runseed $runseed --gnn_type gin --dataset $dataset


### for other GNNs
for gnn_type in gcn gat graphsage
do
python finetune.py --split $split --filename ${dataset}/${gnn_type}_nopretrain --device $device --runseed $runseed --gnn_type $gnn_type --dataset $dataset

model_file=${gnn_type}_supervised_contextpred
python finetune.py --input_model_file model_architecture/${model_file}.pth --split $split --filename ${dataset}/${model_file} --device $device --runseed $runseed --gnn_type $gnn_type --dataset $dataset

done
done


fold_idx=$1

for batch_size in 8 64
do
for drop_ratio in 0 0.2 0.5
do
for dataset in ptc_mr mutag
do
for unsup in contextpred edgepred masking infomax
do

model_file=${unsup}
python finetune_mutag_ptc.py --input_model_file model_gin/${model_file}.pth --dataset $dataset --filename ${dataset}_drop${drop_ratio}_bsize${batch_size}/${model_file} --fold_idx $fold_idx --dropout_ratio $drop_ratio --batch_size $batch_size


model_file=supervised_${unsup}
python finetune_mutag_ptc.py --input_model_file model_gin/${model_file}.pth --dataset $dataset --filename ${dataset}_drop${drop_ratio}_bsize${batch_size}/${model_file} --fold_idx $fold_idx --dropout_ratio $drop_ratio --batch_size $batch_size

done

model_file=supervised
python finetune_mutag_ptc.py --input_model_file model_gin/${model_file}.pth --dataset $dataset --filename ${dataset}_drop${drop_ratio}_bsize${batch_size}/${model_file} --fold_idx $fold_idx --dropout_ratio $drop_ratio --batch_size $batch_size

python finetune_mutag_ptc.py --dataset $dataset --filename ${dataset}_drop${drop_ratio}_bsize${batch_size}/nopretrain --fold_idx $fold_idx --dropout_ratio $drop_ratio --batch_size $batch_size

done
done
done
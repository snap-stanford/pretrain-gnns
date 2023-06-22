#### GIN fine-tuning
runseed=$1
device=$2
split=species

### for GIN
for unsup in contextpred infomax edgepred masking
do
model_file=${unsup}
python finetune.py --model_file model_gin/${model_file}.pth --split $split --filename gin_${model_file} --epochs 50 --device $device --runseed $runseed --gnn_type gin

model_file=supervised_${unsup}
python finetune.py --model_file model_gin/${model_file}.pth --split $split --filename gin_${model_file} --epochs 50 --device $device --runseed $runseed --gnn_type gin
done

python finetune.py --split $split --filename gin_nopretrain --epochs 50 --device $device --runseed $runseed --gnn_type gin
python finetune.py --split $split --model_file model_gin/supervised.pth --filename gin_supervised --epochs 50 --device $device --runseed $runseed --gnn_type gin


### for other GNNs
for gnn_type in gcn gat graphsage
do
python finetune.py --split $split --filename ${gnn_type}_nopretrain --epochs 50 --device $device --runseed $runseed --gnn_type $gnn_type

model_file=${gnn_type}_supervised_masking
python finetune.py --model_file model_architecture/${model_file}.pth --split $split --filename ${model_file} --epochs 50 --device $device --runseed $runseed --gnn_type $gnn_type

done
for seed in 0 1 2 3 
do

# b) Textual Graph + Question
python train.py --dataset webqsp --model_name graph_llm --eval_batch_size 4 --llm_model_name 08b --llm_frozen False --batch_size 2 --seed 1 --warmup_epochs 3 --num_epochs 10 --seed $seed
python train.py --dataset webqsp --model_name graph_llm --eval_batch_size 2 --llm_model_name 2b --llm_frozen False --batch_size 1 --seed 1 --warmup_epochs 3 --num_epochs 10 --seed $seed

done

CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar100 --model lode_esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log &
CUDA_VISIBLE_DEVICES=5 python main.py --dataset seq-cifar100 --model lode_esmer --buffer_size 5120 --load_best_args --seed 1 --csv_log 

# CUDA_VISIBLE_DEVICES=3 python main.py --dataset seq-cifar100 --model lode_esmer --buffer_size 500 --load_best_args --seed 0 --csv_log &
# CUDA_VISIBLE_DEVICES=7 python main.py --dataset seq-cifar100 --model lode_esmer --buffer_size 500 --load_best_args --seed 1 --csv_log 

# seq-CIFAR10
CUDA_VISIBLE_DEVICES=1 python3 main.py --buffer_size (500 5120) --dataset seq-cifar10 --model er --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=1 python3 main.py --buffer_size (500 5120) --dataset seq-cifar10 --model er_lode --load_best_args --seed 0 --csv_log

# seq-CIFAR100
CUDA_VISIBLE_DEVICES=1 python3 main.py --buffer_size (500 5120) --dataset seq-cifar100 --model er --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=1 python3 main.py --buffer_size (500 5120) --dataset seq-cifar100 --model er_lode --load_best_args --seed 0 --csv_log

# seq-tinyimg
CUDA_VISIBLE_DEVICES=1 python3 main.py --buffer_size (500 5120) --dataset seq-tinyimg --model er --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=1 python3 main.py --buffer_size (500 5120) --dataset seq-tinyimg --model er_lode --load_best_args --seed 0 --csv_log


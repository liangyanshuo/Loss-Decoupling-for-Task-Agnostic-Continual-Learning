# seq-cifar10 500, 5120
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar10 --model esmer --buffer_size 500 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar10 --model lode_esmer --buffer_size 500 --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar10 --model esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar10 --model lode_esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log 

# seq-cifar100 500, 5120
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar100 --model esmer --buffer_size 500 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar100 --model lode_esmer --buffer_size 500 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar100 --model esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-cifar100 --model lode_esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log


# seq-tinyimg 500 5120
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-tinyimg --model esmer --buffer_size 500 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-tinyimg --model lode_esmer --buffer_size 500 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-tinyimg --model esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log
CUDA_VISIBLE_DEVICES=2 python main.py --dataset seq-tinyimg --model lode_esmer --buffer_size 5120 --load_best_args --seed 0 --csv_log


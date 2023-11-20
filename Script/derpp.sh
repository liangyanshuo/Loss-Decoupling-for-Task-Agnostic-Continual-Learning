# seq-cifar10 500 5120
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-cifar10 --model derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-cifar10 --model lode_derpp --load_best_args --seed 0 --csv_log 

CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-cifar10 --model derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-cifar10 --model lode_derpp --load_best_args --seed 0 --csv_log 

# seq-cifar100 500 5120
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-cifar100 --model derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-cifar100 --model lode_derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-cifar100 --model derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-cifar100 --model lode_derpp --load_best_args --seed 0 --csv_log 

# seq-tinyimg 500 5120
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-tinyimg --model derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-tinyimg --model lode_derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-tinyimg --model derpp --load_best_args --seed 0 --csv_log 
CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-tinyimg --model lode_derpp --load_best_args --seed 0 --csv_log 

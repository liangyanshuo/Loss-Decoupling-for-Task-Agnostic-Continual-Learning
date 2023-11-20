CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 500 --dataset seq-tinyimg --model derpp --load_best_args --seed 0 --csv_log &
CUDA_VISIBLE_DEVICES=3 python3 main.py --buffer_size 500 --dataset seq-tinyimg --model derpp --load_best_args --seed 1 --csv_log &

CUDA_VISIBLE_DEVICES=5 python3 main.py --buffer_size 500 --dataset seq-tinyimg --model lode_derpp --load_best_args --seed 0 --csv_log &
CUDA_VISIBLE_DEVICES=6 python3 main.py --buffer_size 500 --dataset seq-tinyimg --model lode_derpp --load_best_args --seed 1 --csv_log &

CUDA_VISIBLE_DEVICES=6 python3 main.py --buffer_size 5120 --dataset seq-tinyimg --model derpp --load_best_args --seed 0 --csv_log &
CUDA_VISIBLE_DEVICES=7 python3 main.py --buffer_size 5120 --dataset seq-tinyimg --model derpp --load_best_args --seed 1 --csv_log &

CUDA_VISIBLE_DEVICES=0 python3 main.py --buffer_size 5120 --dataset seq-tinyimg --model lode_derpp --load_best_args --seed 0 --csv_log &
CUDA_VISIBLE_DEVICES=4 python3 main.py --buffer_size 5120 --dataset seq-tinyimg --model lode_derpp --load_best_args --seed 1 --csv_log 


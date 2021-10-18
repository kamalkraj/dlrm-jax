if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi

python dlrm_pytorch.py --arch-sparse-feature-size=16 \
                         --arch-mlp-bot="13-512-256-64-16" \
                         --arch-mlp-top="512-256-1" \
                         --data-generation=dataset \
                         --data-set=kaggle \
                         --raw-data-file=data/train.txt \
                         --loss-function=bce \
                         --round-targets=True \
                         --learning-rate=0.1 \
                         --mini-batch-size=128 \
                         --print-freq=1024 \
                         --print-time \
                         --dataset-multiprocessing \
                         --num-workers=0 \
                         --test-mini-batch-size=16384 \
                         --test-num-workers=16 \
                         --test-freq=10240 \
                         $dlrm_extra_option
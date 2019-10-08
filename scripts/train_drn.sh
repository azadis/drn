python3 segment.py train -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 -s 256 \
	    --arch drn_d_22 --batch-size 32 --epochs 250 --lr 0.01 --momentum 0.9 \
	    --step 100 --ckpt /mnt/disks/sazadi/drn/checkpoints --load_size 512

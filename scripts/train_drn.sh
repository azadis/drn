gpu_ids=1
# CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py train -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 -s 256 \
# 	    --arch drn_d_22 --batch-size 32 --epochs 250 --lr 0.01 --momentum 0.9 \
# 	    --step 100 --ckpt /mnt/disks/sazadi/drn/checkpoints/augmented --load_size 512 --name augmented_


# gpu_ids=0,2,3,4,5,6,7
# CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py train -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 -s 896 \
# 	    --arch drn_d_22 --batch-size 28 --epochs 250 --lr 0.01 --momentum 0.9 --output output_augmented_896\
# 	    --step 100 --ckpt /mnt/disks/sazadi/drn/checkpoints/augmented_896 --load_size 2024 --name augmented_

CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py train -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 -s 256 \
	    --arch drn_d_22 --batch-size 32 --epochs 250 --lr 0.01 --momentum 0.9 \
	    --step 100 --ckpt /mnt/disks/sazadi/drn/checkpoints/augmented_2000 --load_size 512 --name augmented_2000_

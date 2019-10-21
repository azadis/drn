gpu_ids=2
# CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py test -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 --arch drn_d_22 \
#     --resume /mnt/disks/sazadi/drn/checkpoints/drn_d_22/checkpoint_250.pth.tar --phase val --batch-size 1\
#     --load_size 256 --output /mnt/disks/sazadi/drn/output


# CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py test -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 --arch drn_d_22 \
#     --resume /mnt/disks/sazadi/drn/checkpoints/augmented_2000/drn_d_22/checkpoint_250.pth.tar --phase val --batch-size 1\
#     --load_size 256 --output /mnt/disks/sazadi/drn/output/augmented_2000


# CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py test -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 --arch drn_d_22 \
#     --resume /mnt/disks/sazadi/drn/checkpoints/augmented_896/drn_d_22/checkpoint_250.pth.tar --phase val --batch-size 1\
#     --output /mnt/disks/sazadi/drn/output/augmented_896



CUDA_VISIBLE_DEVICES=$gpu_ids python3 segment.py test -d /mnt/disks/sazadi/drn/datasets/cityscapes --name cityscapes_20k -c 19 --arch drn_d_105 \
    --pretrained /mnt/disks/sazadi/drn/checkpoints/drn-d-105_ms_cityscapes.pth --phase test --batch-size 1\
    --output /mnt/disks/sazadi/drn/output/cityscapes_20k 
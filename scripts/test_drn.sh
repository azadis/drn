python3 segment.py test -d /mnt/disks/sazadi/drn/datasets/cityscapes -c 19 --arch drn_d_22 \
    --resume /mnt/disks/sazadi/drn/checkpoints/drn_d_22/checkpoint_003.pth.tar --phase val --batch-size 1\
    --load_size 256 --output /mnt/disks/sazadi/drn/output
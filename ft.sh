# video
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch \
    --nproc_per_node= --nnodes= \
    --node_rank= --master_addr= --master_port= \
    --use_env  \
    --batch_size  --blr\
    --output_dir  \
    --padapt_on 0 --padapt_bottleneck 32 \
    --padapt_L_size 784 --padapt_learn_mode add --padapt_cat_mode x_p \
    --padapt_local a_m_p --padapt_scalar 0.
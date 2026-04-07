CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_blca \
                                      --data_root_dir /data3/share/TCGA/BLCA_feature \
                                      --modal coattn \
                                      --model hdmoe \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_cos \
                                      --lr 0.0005 \
                                      --optimizer Adam \
                                      --scheduler None \
                                      --alpha 1 \
                                      --top_k 1 \
                                      --n_routed_experts 8 \
                                      --gate_dim 256 \
                                      --n_shared_experts True



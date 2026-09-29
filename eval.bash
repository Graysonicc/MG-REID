CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --master_port=25902 train.py \
--config ./configs/retrieval_icfg.yaml \
--output_dir output/test \
--batch_size_test 64 \
--k_test 32 \
--pretrained /output/checkpoint_best.pth \
--evaluate

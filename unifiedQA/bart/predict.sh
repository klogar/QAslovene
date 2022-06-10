data="squad2"
test_bs=8
CUDA_VISIBLE_DEVICES=1 python cli.py --do_predict --output_dir out/${data}_unifiedqa \
        --predict_file data/${data}/dev.tsv \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix dev_
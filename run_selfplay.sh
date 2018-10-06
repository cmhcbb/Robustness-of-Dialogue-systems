CUDA_VISIBLE_DEVICES=2 python adv_selfplay.py --alice_model_file advrl_model_with_agree_ave2.th --bob_model_file rl_model.th --context_file data/negotiate/selfplay.txt --temperature 0.5 --log_file adv_selfplay_with_agree_ave2.log --ref_text data/negotiate/train.txt 


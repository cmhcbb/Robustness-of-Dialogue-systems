CUDA_VISIBLE_DEVICES=2 python adv_selfplay.py --alice_model_file adv_sv_model.th --bob_model_file sv_model.th --context_file data/negotiate/selfplay.txt --temperature 0.5 --log_file adv_vs_sv.log --ref_text data/negotiate/train.txt 


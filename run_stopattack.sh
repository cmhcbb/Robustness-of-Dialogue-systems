CUDA_VISIBLE_DEVICES=1 python stop_attack.py --alice_model_file rl_model.th --bob_model_file rl_model.th --context_file data/negotiate/selfplay.txt --temperature 0.5 --log_file stop_en.log --ref_text data/negotiate/train.txt --max_turns=6


CUDA_VISIBLE_DEVICES=0 python white_attack.py --alice_model_file rl_model.th --bob_model_file rl_model.th --context_file data/negotiate/selfplay.txt --temperature 0.5 --log_file white_alice.log --ref_text data/negotiate/train.txt --max_turns=2


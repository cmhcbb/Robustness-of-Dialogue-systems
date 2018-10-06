CUDA_VISIBLE_DEVICES=3 python adv_reinforce.py   --data data/negotiate   --cuda   --bsz 16   --clip 1   --context_file data/negotiate/selfplay.txt   --eps 0.0   --gamma 0.95   --lr 0.5   --momentum 0.1   --nepoch 4  --nesterov   --ref_text data/negotiate/train.txt   --rl_clip 1   --rl_lr 0.2   --score_threshold 6  --temperature 0.5   --alice_model sv_model.th   --bob_model rl_model.th --output_model_file advrl_model_with_agree_ave2.th --max_turns 20 --seed 2
#--fixed_bob 
#--rl_bob
#--verbose

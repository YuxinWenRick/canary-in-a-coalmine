# Canary in a Coalmine: Better Membership Inference with Ensembled Adversarial Queries

This code is the official implementation of [Canary in a Coalmine](https://arxiv.org/abs/2210.10750).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## Dependencies

- PyTorch => 1.11.*
- torchvision >= 0.12.*

## USAGE
### 1. First, to train shadow models, you can run:
```
bash shadow_models.sh
```

Or, you can create a directory ```saved_models```, and then download our pre-trained shadow models by this [link](https://drive.google.com/drive/folders/15aoIRU7rq4P4FVCxHdWV2UwJxQ7xt7rb?usp=sharing) and put the folder under ```saved_models```.

### 2. Perform attack:
baseline (LiRA):
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_baseline --aug_strategy baseline --num_shadow 64 --num_aug 10 --start 0 --end 5000
```

Canary online:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_online --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
```

Canary offline:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_offline --offline --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --stop_loss 23 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
```

Note: you may disable wandb by adding ```--nowandb```.

For online scenario, you can check ```fix_TPR@0.01FPR``` and ```fix_auc```.

For offline scenario, you can check ```fix_off_TPR@0.01FPR``` and ```fix_off_auc```.

If you want to push ```AUC``` higher (with a slight loss in ```TPR@0.01FPR```) you can try: 

Canary online:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_online --num_shadow 64 --stop_loss 1 --iter 30 --stop_loss 25 --stochastic_k 2 --lr 0.009 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
```
Canary offline:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_offline --offline --num_shadow 64 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --stop_loss 25 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
```

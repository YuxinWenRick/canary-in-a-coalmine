# Canary in a Coalmine: Better Membership Inference with Ensembled Adversarial Queries

This code is the official implementation of [Canary in a Coalmine](paper).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## Dependencies

- PyTorch => 1.11.*
- torchvision >= 0.12.*

## USAGE
### 1. First, to train shadow models, you can run:
```
bash shadow_models.sh
```

### 2. Perform attack:
baseline:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_baseline --aug_strategy baseline --num_shadow 65 --num_aug 10 --no_dataset_aug --start 0 --end 5000
```

Canary online:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_online --num_shadow 65 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
```

Canary offline:
```
python gen_canary.py --name wrn28-10 --save_name wrn28-10_offline --offline --num_shadow 65 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adam --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --stop_loss 25 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
```

Note: you may always disable wandb by adding ```--nowandb```.
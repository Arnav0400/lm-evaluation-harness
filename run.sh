# python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-5.pth.tar --tasks truthfulqa_mc --no_cache --num_fewshot 0 --batch_size 64 --output_path results/llama2-glora4-sharegpt-BS=32/truthfulqa-0shot-ckpt=5.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-5.pth.tar --tasks hendrycksTest-* --no_cache --num_fewshot 5 --batch_size 4 --output_path results/llama2-glora4-sharegpt-BS=32/mmlu-5shot-ckpt=5.json
# python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-5.pth.tar --tasks hellaswag --no_cache --num_fewshot 10 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/hellswag-10shot-ckpt=5.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-5.pth.tar --tasks arc_challenge --no_cache --num_fewshot 25 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/arc-25shot-ckpt=5.json

# python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-10.pth.tar --tasks truthfulqa_mc --no_cache --num_fewshot 0 --batch_size 64 --output_path results/llama2-glora4-sharegpt-BS=32/truthfulqa-0shot-ckpt=10.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-10.pth.tar --tasks hendrycksTest-* --no_cache --num_fewshot 5 --batch_size 4 --output_path results/llama2-glora4-sharegpt-BS=32/mmlu-5shot-ckpt=10.json
# python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-10.pth.tar --tasks hellaswag --no_cache --num_fewshot 10 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/hellswag-10shot-ckpt=10.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-10.pth.tar --tasks arc_challenge --no_cache --num_fewshot 25 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/arc-25shot-ckpt=10.json

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-15.pth.tar --tasks truthfulqa_mc --no_cache --num_fewshot 0 --batch_size 64 --output_path results/llama2-glora4-sharegpt-BS=32/truthfulqa-0shot-ckpt=15.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-15.pth.tar --tasks hendrycksTest-* --no_cache --num_fewshot 5 --batch_size 4 --output_path results/llama2-glora4-sharegpt-BS=32/mmlu-5shot-ckpt=15.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-15.pth.tar --tasks hellaswag --no_cache --num_fewshot 10 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/hellswag-10shot-ckpt=15.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-15.pth.tar --tasks arc_challenge --no_cache --num_fewshot 25 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/arc-25shot-ckpt=15.json

python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-20.pth.tar --tasks truthfulqa_mc --no_cache --num_fewshot 0 --batch_size 64 --output_path results/llama2-glora4-sharegpt-BS=32/truthfulqa-0shot-ckpt=20.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-20.pth.tar --tasks hendrycksTest-* --no_cache --num_fewshot 5 --batch_size 4 --output_path results/llama2-glora4-sharegpt-BS=32/mmlu-5shot-ckpt=20.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-20.pth.tar --tasks hellaswag --no_cache --num_fewshot 10 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/hellswag-10shot-ckpt=20.json
python main.py --model hf-causal-experimental --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4,eval_config=/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-20.pth.tar --tasks arc_challenge --no_cache --num_fewshot 25 --batch_size 8 --output_path results/llama2-glora4-sharegpt-BS=32/arc-25shot-ckpt=20.json

# python main.py --model hf-causal-experimental --model_args pretrained=Arnav0400/llama-glora,dtype='float16' --tasks truthfulqa_mc --no_cache --num_fewshot 0 --batch_size 64 --output_path results/glora4_merged_shareGPT/truthfulqa-0shot-ckpt=5.json
# python main.py --model hf-causal-experimental --model_args pretrained=Arnav0400/llama-glora,dtype='float16' --tasks hendrycksTest-* --no_cache --num_fewshot 5 --batch_size auto --output_path results/glora4_merged_shareGPT/mmlu-5shot-ckpt=5.json
# python main.py --model hf-causal-experimental --model_args pretrained=Arnav0400/llama-glora,dtype='float16' --tasks hellaswag --no_cache --num_fewshot 10 --batch_size 16 --output_path results/glora4_merged_shareGPT/hellswag-10shot-ckpt=5.json
# python main.py --model hf-causal-experimental --model_args pretrained=Arnav0400/llama-glora,dtype='float16' --tasks arc_challenge --no_cache --num_fewshot 25 --batch_size 8 --output_path results/glora4_merged_shareGPT/arc-25shot-ckpt=5.json
gpu_id=0
gen_model_name=google/mt5-large
cls_model_name=roberta-large-mnli

for mode in E_only
do
    python src/t5_summarize.py \
    --gpu_id $gpu_id \
    --data_path data/pa_test.txt \
    --batch_size 4 \
    --gen_model_name $gen_model_name \
    --cls_model_name $cls_model_name \
    --decoding_strategy 'top-k' \
    --topk_value 40 \
    --num_generate_per_sentence 10 \
    --filter_mode $mode \
    --output_length 128 \
    --output_tokens
done

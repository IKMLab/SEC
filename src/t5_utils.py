from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import torch
import torch.nn as nn


def load_generation_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model.to(device)


def load_filtering_model(model_name, device):
    NLI_tokenizer = AutoTokenizer.from_pretrained(model_name)
    NLI_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return NLI_tokenizer, NLI_model.to(device)


def t5_generate(
    ori_sentences,
    batch_size,
    gen_model_name,
    num_generate_per_sentence,
    decoding_strategy,
    decoding_value,
    output_length,
    device,
    save_seqs_name,
):
    t5_tokenizer, t5_model = load_generation_model(
        gen_model_name,
        device,
    )
    do_sample = True if decoding_strategy in ['top-k', 'top-p'] else False

    generated_sentences = []
    for i in tqdm(range(0, len(ori_sentences), batch_size)):
        batch_sentences = ori_sentences[i:i+batch_size]
        batch = t5_tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids
        batch = batch.to(device)

        with torch.no_grad():
            if decoding_strategy == 'top-k':
                model_output = t5_model.generate(
                    batch,
                    top_k=decoding_value,
                    max_length=output_length,
                    do_sample=do_sample,
                    num_return_sequences=num_generate_per_sentence,
                )
            elif decoding_strategy == 'top-p':
                t5_model.config.top_k = None
                model_output = t5_model.generate(
                    batch,
                    max_length=output_length,
                    top_p=decoding_value,
                    do_sample=do_sample,
                    num_return_sequences=num_generate_per_sentence,
                )
            elif decoding_strategy == 'beam':
                model_output = t5_model.generate(
                    batch,
                    max_length=output_length,
                    num_beams=decoding_value,
                    do_sample=do_sample,
                    num_return_sequences=num_generate_per_sentence,
                )

            result = t5_tokenizer.batch_decode(
                model_output, skip_special_tokens=True)
            generated_sentences.extend(result)

    print("Finished generating data.")
    del t5_tokenizer
    del t5_model

    with open(save_seqs_name, 'w') as f:
        for i in range(len(generated_sentences)):
            f.write(generated_sentences[i] + '\n')

    return generated_sentences


def sentence_filtering(
    origin_sentence,
    new_sentence,
    tokenizer,
    model,
    filter_mode,
    device,
):
    data_encoding = tokenizer(
        origin_sentence,
        new_sentence,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**data_encoding).logits
        predict_proba = nn.functional.softmax(outputs, dim=-1)

        # 觀察是否有任何一個 sentence 會被預測為 2 (entailment)
        # roberta-large-mnli: {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}
        predictions = predict_proba.argmax(dim=-1)

        if filter_mode == 'E_only':
            got_passed = torch.any(predictions == 2).cpu().numpy()
            best_choice = predict_proba[:, 2].argmax().item()

        elif filter_mode == 'EorN':
            got_passed = torch.any(predictions != 0).cpu().numpy()
            if got_passed:
                if torch.any(predictions == 2).cpu().numpy():
                    best_choice = predict_proba[:, 2].argmax().item()
                else:
                    best_choice = predict_proba[:, 1].argmax().item()
            elif not got_passed:
                best_choice = -1
                print(f"Ori: {origin_sentence[0]}")
                print(f"Wrong: {new_sentence[predict_proba[:, 2].argmax()]}")

        elif filter_mode == 'C_only':
            got_passed = torch.any(predictions == 0).cpu().numpy()
            best_choice = predict_proba[:, 0].argmax().item()

        elif filter_mode == 'N_only':
            got_passed = torch.any(predictions == 1).cpu().numpy()
            best_choice = predict_proba[:, 1].argmax().item()

    return best_choice, got_passed

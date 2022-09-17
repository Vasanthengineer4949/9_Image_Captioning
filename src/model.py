import config
from dataset import load_dataset
import pandas as pd
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel, ViTFeatureExtractor, DefaultDataCollator

captions_path = config.CAPTION_PATH
img_dir = config.IMG_PATH
encoder = config.ENCODER_CKPT
decoder = config.DECODER_CKPT
model_out_dir = config.MODEL_SAVE_DIR

feature_extractor = ViTFeatureExtractor.from_pretrained(encoder)
tokenizer = AutoTokenizer.from_pretrained(decoder)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

with open(captions_path) as f:
    data = []
    for i in f.readlines():
        splitted_text = i.split(",")
        data.append([splitted_text[0], " ".join(splitted_text[1:])])

df = pd.DataFrame(data, columns=["images", "text"])
print(df)

train_dataset = load_dataset(img_dir, df, feature_extractor, tokenizer)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder, decoder
)

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = config.MAX_LENGTH
model.config.early_stopping = config.EARLY_STOPPING
model.config.no_repeat_ngram_size = config.N_GRAMS
model.config.length_penalty = 2.0
model.config.num_beams = config.NUM_BEAMS
model.decoder.resize_token_embeddings(len(tokenizer))

for param in model.encoder.parameters():
    param.requires_grad = False


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="no",
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    overwrite_output_dir=True,
    fp16=True,
    run_name="first_run",
    output_dir=model_out_dir,
    logging_steps=config.LOGGING_STEPS,
    save_steps = config.SAVE_STEPS,
    num_train_epochs=config.NUM_EPOCHS,
    push_to_hub=True
)


if __name__ == "__main__":
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        data_collator=DefaultDataCollator(),
    )
    trainer.train()
    trainer.push_to_hub()
import sentencepiece as spm

TRAINING_CORPUS = "./data/wikitext103.csv"

TEXT = "Once upon a time there lived in a certain village a little country girl, " + \
       "the prettiest creature who was ever seen. Her mother was excessively fond of her; " + \
       "and her grandmother doted on her still more. This good woman had a little red riding hood made for her. " + \
       "It suited the girl so extremely well that everybody called her Little Red Riding Hood."

MODEL_OUTPUT_PREFIX = "./model_data/sentence_piece"
VOCAB_SIZE = 16000

train = f"--input={TRAINING_CORPUS} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_prefix={MODEL_OUTPUT_PREFIX} --vocab_size={VOCAB_SIZE} --model_type=bpe"

spm.SentencePieceTrainer.Train(train)

#%%
sp = spm.SentencePieceProcessor()
sp.Load(f"{MODEL_OUTPUT_PREFIX}.model")

pieces = sp.EncodeAsPieces(TEXT)

print("[Encoded text]")
print("-" * 50)
print(pieces[:10])

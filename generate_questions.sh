DATA_DIR=data/test
MODEL_RECOVER_PATH=pretrained_models/qg_model.bin
EVAL_SPLIT=test
#export PYTORCH_PRETRAINED_BERT_CACHE=/home/osainz/.pytorch_pretrained_bert/cee054f6aafe5e2cf816d2228704e326446785f940f5451a5b26033516a4ac3d.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1
# run decoding
python3 biunilm/decode_seq2seq.py --bert_model bert-large-cased --new_segment_ids --mode s2s \
  --input_file ${DATA_DIR}/test_3.pa.txt --split ${EVAL_SPLIT} \
  --output_file ${DATA_DIR}/test_3.output.pa.tok.txt \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 512 --max_tgt_length 48 \
  --batch_size 16 --beam_size 1 --length_penalty 0
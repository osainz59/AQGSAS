import torch
import random

import numpy as np
from tqdm import tqdm

from .pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from .pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from .pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from .seq2seq_loader import *

SEED = 123

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)

class QuestionGenerator:

    def __init__(self, model_recover_path, bert_model="bert-large-cased", do_lower_case=False, 
                 max_seq_length=512, max_tgt_length=128, new_segment_ids=True, mode="s2s", num_qkv=0,
                 s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False,
                 pos_shift=False, fp16=False, amp=False, forbid_duplicate_ngrams=False,
                 forbid_ignore_word=None, not_predict_token=None, beam_size=1,
                 length_penalty=0., ngram_size=3, min_len=None, ffn_type=0,
                 seg_emb=False, batch_size=16):

        self.model_recover_path = model_recover_path
        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.max_tgt_length = max_tgt_length
        self.new_segment_ids = new_segment_ids
        self.mode = mode
        assert self.mode in ["s2s", "l2r", "both"]
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.fp16 = fp16
        self.amp = amp
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_word = forbid_ignore_word
        self.not_predict_token = not_predict_token
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.ngram_size = ngram_size
        self.min_len = min_len
        self.ffn_type = ffn_type
        self.seg_emb = seg_emb
        self.batch_size = batch_size

        #######################################
        self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(SEED)

        self.tokenizer = BertTokenizer.from_pretrained(
            self.bert_model, do_lower_case=self.do_lower_case)

        self.tokenizer.max_len = self.max_seq_length

        pair_num_relation = 0
        self.bi_uni_pipeline = []
        self.bi_uni_pipeline.append(Preprocess4Seq2seqDecoder(list(self.tokenizer.vocab.keys()), 
                                self.tokenizer.convert_tokens_to_ids, self.max_seq_length, 
                                max_tgt_length=self.max_tgt_length, new_segment_ids=self.new_segment_ids,
                                mode=self.mode, num_qkv=self.num_qkv, s2s_special_token=self.s2s_special_token, 
                                s2s_add_segment=self.s2s_add_segment, s2s_share_segment=self.s2s_share_segment, 
                                pos_shift=self.pos_shift))

        amp_handle = None
        if self.fp16 and self.amp:
            from apex import amp
            amp_handle = amp.init(enable_caching=True)

        # Prepare model
        cls_num_labels = 2
        type_vocab_size = 6 + \
            (1 if self.s2s_add_segment else 0) if self.new_segment_ids else 2
        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(
            ["[MASK]", "[SEP]", "[S2S_SOS]"])

        def _get_token_id_set(s):
            r = None
            if s:
                w_list = []
                for w in s.split('|'):
                    if w.startswith('[') and w.endswith(']'):
                        w_list.append(w.upper())
                    else:
                        w_list.append(w)
                r = set(self.tokenizer.convert_tokens_to_ids(w_list))
            return r

        forbid_ignore_set = _get_token_id_set(self.forbid_ignore_word)
        not_predict_set = _get_token_id_set(self.not_predict_token)

        model_recover_path = self.model_recover_path
        print("***** Recover model: %s *****" % model_recover_path)
        model_recover = torch.load(model_recover_path, map_location=self.device)
        self.model = BertForSeq2SeqDecoder.from_pretrained(self.bert_model, state_dict=model_recover, num_labels=cls_num_labels, 
                                num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, 
                                search_beam_size=self.beam_size, length_penalty=self.length_penalty, eos_id=eos_word_ids, 
                                sos_id=sos_word_id, forbid_duplicate_ngrams=self.forbid_duplicate_ngrams, 
                                forbid_ignore_set=forbid_ignore_set, not_predict_set=not_predict_set, ngram_size=self.ngram_size, 
                                min_len=self.min_len, mode=self.mode, max_position_embeddings=self.max_seq_length, 
                                ffn_type=self.ffn_type, num_qkv=self.num_qkv, seg_emb=self.seg_emb, pos_shift=self.pos_shift)
        del model_recover

        if self.fp16:
            self.model.half()
        self.model.to(self.device)
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        torch.cuda.empty_cache()
        self.model.eval()

    def generate_questions_from_text(self, text):
        input_lines = [x.strip() for x in text]

        next_i = 0
        max_src_length = self.max_seq_length - 2 - self.max_tgt_length

        #data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
        input_lines = [self.tokenizer.tokenize(
            x)[:max_src_length] for x in input_lines]
        input_lines = sorted(list(enumerate(input_lines)),
                                key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / self.batch_size)

        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + self.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += self.batch_size
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in self.bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(instances)
                    batch = [
                        t.to(self.device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = self.model(input_ids, token_type_ids,
                                    position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    if self.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_lines[buf_id[i]] = output_sequence
                pbar.update(1)

        del traces
        torch.cuda.empty_cache()
        return output_lines
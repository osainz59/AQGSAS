import re
from flair.models import SequenceTagger
from flair.embeddings import Sentence


class Vocabulary(set):

    def __init__(self, word_list, model='ner-ontonotes'):
        super(Vocabulary, self).__init__(word_list)
        self.compiled = None
        self.ner_tagger = SequenceTagger.load(model) if model else None

    @classmethod
    def from_vocab_file(cls, file_path, **kwargs):
        with open(file_path) as in_f:
            return cls((word.rstrip() for word in in_f), **kwargs)

    def to_vocab_file(self, file_path, encoding='utf-8'):
        with open(file_path, encoding=encoding) as out_f:
            for word in self:
                out_f.write(word + '\n')

    def __repr__(self):
        return f"(Vocabulary) - size: {len(self)} - compiled: {self.compiled}"

    def compile(self):
        word_list_sorted_by_length = sorted(list(self), key=lambda x: len(x), reverse=True)
        self.compiled = re.compile("|".join(word_list_sorted_by_length), flags=re.IGNORECASE)

        return self

    def find(self, text):
        if not self.compiled:
            raise Exception('You need to compile the vocabulary first.')

        # Apply the general tagger
        if self.ner_tagger:
            text_ = Sentence(text)
            self.ner_tagger.predict(text_)
            for ent in text_.to_dict(tag_type='ner')['entities']:
                yield {'text': ent['text'], 'start_pos': ent['start_pos'], 'end_pos': ent['end_pos']}

        # Apply the especialized vocabulary
        if self.compiled:
            for item in self.compiled.finditer(text):
                span = item.span(0)
                text = item.group(0)
                yield {'text': text, 'start_pos': span[0], 'end_pos': span[1]}



def test():
    vocab = Vocabulary(['a', 'b', 'c'], model=None)
    print(vocab)
    vocab = Vocabulary.from_vocab_file('vocabularies/biology.vocab')
    vocab.compile()
    print(vocab)
    print(list(vocab.find("There are a lot of dwarf in the benthos.")))

if __name__ == "__main__":
    test()

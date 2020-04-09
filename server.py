from flask import Flask, render_template, request, Response, session
import json
import re

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
from .biunilm.question_generator import QuestionGenerator
from .vocabulary import Vocabulary
from transformers import pipeline

import torch

# Some parameters
SCORE_THRESHOLD = .6        # [0., 1.] The minimum value for an aceptable question
WINDOW_LENGTH = 128         # Character window length
QG_BEAM_SIZE = 3            # Beam-size used on question generation decoder

# 
# tagger = SequenceTagger.load('ner-ontonotes')
ne = Vocabulary.from_vocab_file('vocabularies/biology.vocab').compile()
qg = QuestionGenerator('pretrained_models/qg_model.bin', beam_size=QG_BEAM_SIZE)
qa = pipeline('question-answering')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([WordEmbeddings('glove'),
                                              FlairEmbeddings('news-backward'),
                                              FlairEmbeddings('news-forward')])

remove_punct = re.compile(r"[\(\)\'\":?¿!¡;]")


def answer_similarity(ans1, real):
    sent1 = Sentence(ans1)
    sent2 = Sentence(real)
    document_embeddings.embed(sent1)
    document_embeddings.embed(sent2)
    emb1 = sent1.get_embedding()
    emb2 = sent2.get_embedding()
    emb1 /= torch.sqrt((emb1**2).sum())
    emb2 /= torch.sqrt((emb2**2).sum())

    return max(0., (emb1.T @ emb2).item())

def is_a_good_question(question, answer, score):
    return score > SCORE_THRESHOLD and 'SEP' not in question

app = Flask(__name__)
app.secret_key = "imnotsecretatall"

@app.route("/")
def index():
    return render_template("demo.html")

@app.route("/correct_answers", methods=["POST"])
def correct_answers():
    if request.method == 'POST':
        idx = request.get_json()['id']
        answer = request.get_json()['answer']
        
        if answer == "":
            return Response("{'a': 'b'}", status=204, mimetype='application/json')

        correct_answer = session['questions'][idx]['answer']
        correct_answer_confidence = session['questions'][idx]['score']

        score = answer_similarity(answer, correct_answer)

        return json.dumps({
            'correct_answer': correct_answer,
            'correct_answer_confidence': correct_answer_confidence,
            'score': score
        })

    else:
        return Response("{'a': 'b'}", status=204, mimetype='application/json')


@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    if request.method == 'POST':
        # Clear previous session
        session.clear()

        # Get the text
        text = request.get_json()['text'].replace('\n', ' ')

        # Extract the Named Entities
        #sentence = Sentence(text.replace(',','').replace('-','').replace('(','').replace(')', '').replace('\n', ' '))
        entities, ent_pos = [], []
        for ent in ne.find(text):
            entities.append(ent['text'])
            ent_pos.append((ent['start_pos'], ent['end_pos']))
        #entities = set(ent['text'] for ent in sentence.to_dict(tag_type='ner')['entities'])

        text_fragments = [
            text[max(0, start_pos-WINDOW_LENGTH):end_pos+WINDOW_LENGTH] for start_pos, end_pos in ent_pos
        ]

        if len(entities) == 0:
            return json.dumps([])

        # Third, we generate the posible text-answer candidates
        candidates = [
            tf + ' [SEP]' + entity for entity, tf in zip(entities, text_fragments)
        ]

        def remove_repeated(questions):
            q, idx = set(), []
            for i, question in enumerate(questions):
                if question not in q:
                    q.add(question)
                    idx.append(i)

            return q, idx

        # Finally we extract our questions
        questions, tf_ids = remove_repeated(qg.generate_questions_from_text(candidates))
        text_fragments = [text_fragments[i] for i in tf_ids]

        # Generate new answers for the questions
        answers, scores = [], []
        def get_answers(questions, text):
            """ Support function to get always a list as a result
            """
            answers = qa([{'question': question,'context': tf} for question, tf in zip(questions, text)])
            if isinstance(answers, dict):
                answers = [answers]
            return answers

        for ans in get_answers(questions, text_fragments):
            answers.append(remove_punct.sub("", ans['answer']).strip(', '))
            scores.append(ans['score'])
        
        response = []
        if not 'questions' in session:
            session['questions'] = []

        idx = 0
        for question, answer, score in zip(questions, answers, scores):
            if is_a_good_question(question, answer, score):
                response.append({
                    'question': question,
                    'answer': answer,
                    'score': score,
                    'id': idx
                })
                session['questions'].append({
                    'question': question,
                    'answer': answer,
                    'score': score
                })
                idx += 1


        return json.dumps(response)
    else:
        return Response("{'a': 'b'}", status=204, mimetype='application/json')
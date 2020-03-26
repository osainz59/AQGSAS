from flask import Flask, render_template, request, Response, session
import json
import re

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
from flair.models import SequenceTagger
from .biunilm.question_generator import QuestionGenerator
from transformers import pipeline

import torch

# Some parameters
SCORE_THRESHOLD = .6        # [0., 1.] The minimum value for an aceptable question


tagger = SequenceTagger.load('ner-ontonotes')
qg = QuestionGenerator('pretrained_models/qg_model.bin')
qa = pipeline('question-answering')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([WordEmbeddings('glove'),
                                              FlairEmbeddings('news-backward'),
                                              FlairEmbeddings('news-forward')])

remove_punct = re.compile(r"[,\.\'\":?¿!¡;]")


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
        sentence = Sentence(text)
        tagger.predict(sentence)
        entities = set(ent['text'] for ent in sentence.to_dict(tag_type='ner')['entities'])

        if len(entities) == 0:
            return json.dumps([])

        # Third, we generate the posible text-answer candidates
        candidates = [
            text + ' [SEP]' + entity for entity in entities
        ]

        # Finally we extract our questions
        questions = set(qg.generate_questions_from_text(candidates))

        # Generate new answers for the questions
        answers, scores = [], []
        def get_answers(questions, text):
            """ Support function to get always a list as a result
            """
            answers = qa([{'question': question,'context': text} for question in questions])
            if isinstance(answers, dict):
                answers = [answers]
            return answers

        for ans in get_answers(questions, text):
            print(ans)
            answers.append(remove_punct.sub("", ans['answer']))
            scores.append(ans['score'])
        
        response = []
        if not 'questions' in session:
            session['questions'] = []

        for idx, (question, answer, score) in enumerate(zip(questions, answers, scores)):
            if score > SCORE_THRESHOLD:
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


        return json.dumps(response)
    else:
        return Response("{'a': 'b'}", status=204, mimetype='application/json')
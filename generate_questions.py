from flair.data import Sentence
from flair.models import SequenceTagger

from pprint import pprint

from biunilm.question_generator import QuestionGenerator
from rgem.framework import RGEMFramework

from transformers import pipeline


def ner_qg():
    # First we select our text paragraph
    text = """For a lot of people in the US, there's an additional worry: how much a coronavirus test might cost.
In a tweet posted two days ago and "liked" more than 230,000 times, one American claims to have found a "loophole".
"If you don't have insurance and can't afford to take the $3,200 test for the virus ($1,000 with insurance), DONATE BLOOD," she writes.
"They HAVE to test you for the virus in order to donate blood."
The figures she cites are based on another viral social media post, about a man who reportedly returned to Florida from China with flu symptoms, that was later debunked. 
Because the US has a largely private healthcare system, the cost of a coronavirus test is largely up to what kind of insurance a person has, and which particular lab carries out the test.
Experts are concerned that people with poor cover or no insurance will avoid getting tested - leaving them open to serious illness, and to infecting others.
US Vice President Mike Pence has said cost will not be a barrier to people getting tested, because Medicare and Medicaid healthcare payment systems will cover the costs of those who need government healhcare support.
A group of major health insurance companies also said there would be no surprise billing for costs associated with the coronavirus."""

    # Second, we extract the posible interesting entities based on Ontonotes
    sentence = Sentence(text.replace(',','').replace('-','').replace('(','').replace(')', '').replace('\n', ' '))
    tagger = SequenceTagger.load('ner-ontonotes')
    tagger.predict(sentence)
    entities = set(ent['text'] for ent in sentence.to_dict(tag_type='ner')['entities'])

    # Third, we generate the posible text-answer candidates
    candidates = [
        text + ' [SEP]' + entity for entity in list(entities)
    ]

    # We load our question generation system
    qg = QuestionGenerator('pretrained_models/qg_model.bin')

    # Finally we extract our questions
    questions = qg.generate_questions_from_text(candidates)

    # And validate them with a question answering system
    nlp = pipeline('question-answering')
    answers = [
        f"{ans['answer']} ({ans['score']})" for ans in nlp([
            {
                'question': question,
                'context': text
            } for question in questions
        ])
    ]

    for question, ent, answer in zip(questions, list(entities), answers):
        print(f"{question} {ent} / {answer}")


if __name__ == "__main__":
    ner_qg()
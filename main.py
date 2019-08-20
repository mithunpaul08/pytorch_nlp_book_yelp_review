from model.classifier import ReviewClassifier
from modules.dataset import ReviewDataset
from set_parameters import Initializer
from train import Trainer
from eval import Evaluator


Initializer.set_parameters()
dataset=Initializer.read_data_make_vectorizer()

vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
train_state=Trainer.train(classifier,dataset)
Evaluator.evaluate(classifier,data,train_state)
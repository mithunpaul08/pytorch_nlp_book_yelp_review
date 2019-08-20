

class TestYelpReviewPsentiment():
    def test_model_load(self):
        train_state = make_train_state(args)
        classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
        print("coming here at 1")
        classifier.load_state_dict(torch.load(train_state['model_filename']))
        print(f"value of train_state['model_filename']is:{train_state['model_filename']}")
        classifier = classifier.cpu()
        print("coming here at 2")
        assert isinstance(classifier, ReviewClassifier)
        import sys
        sys.exit(1)
test_model_load(self)
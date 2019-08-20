from utils.utils import generate_batches

class Evaluator:
    def evaluate(self,classifier,dataset,train_state):
        # compute the loss & accuracy on the test set using the best available model

        classifier.load_state_dict(torch.load(train_state['model_filename']))
        classifier = classifier.to(args.device)

        dataset.set_split('test')
        batch_generator = generate_batches(dataset,
                                           batch_size=args.batch_size,
                                           device=args.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred = classifier(x_in=batch_dict['x_data'].float())

            # compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['test_loss'] = running_loss
        train_state['test_acc'] = running_acc
        print("Test loss: {:.3f}".format(train_state['test_loss']))
        print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
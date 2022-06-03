import os
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn import metrics
import numpy as np
from torch import no_grad, save as torch_save
from torch.utils.tensorboard import SummaryWriter

def dump_metrics(
    metrics: dict, writer: SummaryWriter, prefix: str, global_step: int
):
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f'{metric_name}/{prefix}',
            metric_value,
            global_step=global_step,
        )

class Trainer:
    def __init__(
        self,
        model,
        output_path,
        writer,
        optimizer=None,
        criterion=None,
        epochs=40,
        lr=0.1,
    ):
        if optimizer is None:
            optimizer = SGD(model.parameters(), lr=lr)
        self.optimizer = optimizer
        if criterion is None:
            criterion = CrossEntropyLoss()
        self.criterion = criterion
        self.epochs = epochs
        self.model = model
        self.output_path = output_path
        self.writer = writer

    def train_step(self, inputs, labels):
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def val_step(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def train(self, dataloader, validloader, device, log_step=100):
        self.model.train()
        min_valid_loss = np.inf

        for epoch in range(self.epochs):
            mean_loss_per_ep = 0
            running_loss = 0.0
            made_steps = epoch * len(dataloader)

            for i, data in enumerate(dataloader):
                metrics = dict()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                loss = self.train_step(inputs, labels)
                metrics['loss'] = loss.item().cpu()

                # print statistics
                running_loss += loss.item()
                mean_loss_per_ep += loss.item()
                if i % log_step == (log_step - 1):
                    print(
                        '[%d, %5d] loss: %.3f'
                        % (epoch + 1, i + 1, running_loss / log_step)
                    )
                    running_loss = 0.0
                dump_metrics(metrics, self.writer, prefix='train', global_step=made_steps + i)

            with no_grad():
                valid_loss = 0.0
                for j, data in enumerate(validloader):
                    metrics = dict()
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    loss = self.val_step(inputs, labels)
                    metrics['loss'] = loss.item().cpu()
                    valid_loss += loss.item()
                    dump_metrics(metrics, self.writer, prefix='valid', global_step=made_steps + j)

                if min_valid_loss > valid_loss:
                    min_valid_loss = valid_loss
                    model_path = os.path.join(
                        self.output_path, 'best_model.pth'
                    )
                    torch_save(self.model.state_dict(), model_path)

            print(
                f'Mean training loss per epoch {epoch + 1} = ',
                f'{mean_loss_per_ep / (i + 1)}\n',
                f'Mean validation loss per epoch {epoch + 1} = ',
                f'{valid_loss / (j + 1)}',
            )
        print('Finished Training')

    def test(self, dataloader, device):
        self.model.eval()
        preds = np.empty(0)
        true_lab = np.empty(0)
        with no_grad():

            for im, la in dataloader:
                im = im.to(device)
                pred = self.model(im)
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                preds = np.append(preds, pred)
                true_lab = np.append(true_lab, la.cpu().detach().numpy())

        acc = metrics.accuracy_score(true_lab, preds)
        return acc

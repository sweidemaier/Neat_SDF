
class BaseTrainer():

    def __init__(self, cfg):
        pass

    def update(self):
        raise NotImplementedError("Trainer [update] not implemented.")

    def log_train(self, train_info, train_data,
                  writer=None, step=None, epoch=None, visualize=False,
                  ):
        raise NotImplementedError("Trainer [log_train] not implemented.")

    def validate(self, test_loader, epoch):
        raise NotImplementedError("Trainer [validate] not implemented.")

    def log_val(self, val_info, writer=None, step=None, epoch=None):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    writer.add_scalar(k, v, step)
                else:
                    writer.add_scalar(k, v, epoch)

    def save(self, epoch=None, step=None, appendix=None):
        raise NotImplementedError("Trainer [save] not implemented.")

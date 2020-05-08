from methods.baseMethod import baseMethod
import torch

class EDSR_training_method(baseMethod):
    def __init__(self, args, loader, my_model, my_loss, ckp, log_writer):
        super(EDSR_training_method, self).__init__(args, loader, my_model, my_loss, ckp, log_writer)

        if args.freeze_initial_layers:
            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False
                if i == 34:
                    break

        # TensorBoard model graph log
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
        test_loader = loader.loader_test
        lr_batch, hr_batch = next(iter(test_loader))
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
        log_writer.add_graph(my_model, lr_batch)
        log_writer.close()



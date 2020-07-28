from methods.baseMethod import baseMethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoderDecoder_method(baseMethod):
    def __init__(self, args, loader, my_model, my_loss, ckp, log_writer):
        super(encoderDecoder_method, self).__init__(args, loader, my_model, my_loss, ckp, log_writer)

        if args.fine_tuning:
            args.resume  = 0

        # TensorBoard model graph log
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
        test_loader = loader.loader_test
        lr_batch, _ = next(iter(test_loader))
        lr_batch = lr_batch.to(device)
        log_writer.add_graph(my_model, lr_batch)
        log_writer.close()

    def train_batch(self, lr, hr, **kwargs):
        assert 'hr_eo' in kwargs, 'Visible spectrum image does not exist'

        self.optimizer.zero_grad()
        sr_ir, eo = self.model(lr, kwargs['hr_eo'])
        loss_ir = self.loss(sr_ir, hr)
        loss_eo = self.loss(eo, kwargs['hr_eo'])
        losses = []
        losses.append(loss_eo)
        losses.append(loss_ir)
        losses.append(loss_eo + loss_ir)
        loss = sum(losses[:-1])
        loss.backward()
        for p in self.model.encoder.encoder_shared.parameters():
            p.grad.data = 0.5 * p.grad.data
        for p in self.model.decoder.decoder_shared.parameters():
            p.grad.data = 0.5 * p.grad.data

        if self.args.gclip > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.args.gclip
            )

        self.optimizer.step()

        return losses, sr_ir

    def test_batch(self, lr, hr=None, evaluation=False, interpolate=False, **kwargs):
        assert 'hr_eo' in kwargs, 'Visible spectrum image does not exist'

        if interpolate:
            lr_batch = [torch.nn.functional.interpolate(lr[i, ...].unsqueeze(0),
                                                        scale_factor=self.scale,
                                                        mode='bicubic', align_corners=True).squeeze() for i in
                        range(lr.shape[0])]
            result = torch.stack(lr_batch, dim=0)
            eo = kwargs['hr_eo']
        else:
            sr_ir, eo = self.model(lr, kwargs['hr_eo'])

        if evaluation:
            try:
                loss_ir = self.loss(sr_ir, hr)
                loss_eo = self.loss(eo, kwargs['hr_eo'])
                losses = []
                losses.append(loss_eo)
                losses.append(loss_ir)
                losses.append(loss_eo + loss_ir)
                loss_types = ['loss_eo', 'loss_ir', 'loss_total']
                loss = sum(losses[:-1])
            except:
                # print("Cannot calculate loss for this method")
                losses, loss_types = [torch.zeros(1) for _ in self.loss_types], self.loss_types

            result = sr_ir, eo, losses, loss_types
        return result

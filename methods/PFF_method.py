from methods.baseMethod import baseMethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class PFF_method(baseMethod):
    def __init__(self, args, loader, my_model, my_loss, ckp, log_writer):
        super(PFF_method, self).__init__(args, loader, my_model, my_loss, ckp, log_writer)

        self.filterSize = args.filterSize

        if args.fine_tuning:
            args.resume  = 0
        # self.model.load(
        #     ckp.get_path('model'),
        #     pre_train=args.pre_train,
        #     resume=args.resume,
        # )
        if args.freeze_initial_layers:
            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False
                if i == 34:
                    break
        self.scale = 1

        # TensorBoard model graph log
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
        test_loader = loader.loader_test
        lr_batch, hr_batch = next(iter(test_loader))
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
        log_writer.add_graph(my_model, lr_batch)
        log_writer.close()

    def train_batch(self, lr, hr):
        self.optimizer.zero_grad()
        embed = self.model(lr, 0)
        losses, loss_types = self.loss(embed, hr, lr=lr)
        loss = sum(losses)
        loss.backward()
        if self.args.gclip > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.args.gclip
            )
        self.optimizer.step()

        return losses, embed

    def test_batch(self, lr, hr=None, evaluation=False, interpolate=False):
        if interpolate:
            result = lr
        else:
            embedding1_to_2 = self.model(lr, 0)
            result = self.rgbImageFilterFlow(lr, embedding1_to_2)
        if evaluation:
            try:
                losses, loss_types = self.loss(embedding1_to_2, hr, lr=lr, type='test')
            except:
                # print("Cannot calculate loss for this method")
                losses, loss_types = [torch.zeros(1) for _ in self.loss_types], self.loss_types
            result = result, losses, loss_types
        return result

    def rgbImageFilterFlow(self, img, filters):
        inputChannelSize = 1
        outputChannelSize = 1
        N = img.size(0)
        paddingFunc = nn.ZeroPad2d(int(self.filterSize / 2))
        img = paddingFunc(img)
        imgSize = [img.size(2), img.size(3)]

        out_R = F.unfold(img[:, 0, :, :].unsqueeze(1), (self.filterSize, self.filterSize))
        out_R = out_R.view(N, out_R.size(1), imgSize[0] - self.filterSize + 1, imgSize[1] - self.filterSize + 1)
        # out_R = paddingFunc(out_R)
        out_R = torch.mul(out_R, filters)
        out_R = torch.sum(out_R, dim=1).unsqueeze(1)

        out_G = F.unfold(img[:, 1, :, :].unsqueeze(1), (self.filterSize, self.filterSize))
        out_G = out_G.view(N, out_G.size(1), imgSize[0] - self.filterSize + 1, imgSize[1] - self.filterSize + 1)
        # out_G = paddingFunc(out_G)
        out_G = torch.mul(out_G, filters)
        out_G = torch.sum(out_G, dim=1).unsqueeze(1)

        out_B = F.unfold(img[:, 2, :, :].unsqueeze(1), (self.filterSize, self.filterSize))
        out_B = out_B.view(N, out_B.size(1), imgSize[0] - self.filterSize + 1, imgSize[1] - self.filterSize + 1)
        # out_B = paddingFunc(out_B)
        out_B = torch.mul(out_B, filters)
        out_B = torch.sum(out_B, dim=1).unsqueeze(1)
        return torch.cat([out_R, out_G, out_B], 1)



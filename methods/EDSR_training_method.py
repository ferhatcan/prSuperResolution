from methods.baseMethod import baseMethod

class EDSR_training_method(baseMethod):
    def __init__(self, args, loader, my_model, my_loss, ckp, log_writer):
        super(EDSR_training_method, self).__init__(args, loader, my_model, my_loss, ckp, log_writer)

        if args.fine_tuning:
            args.resume  = 0
        self.model.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
        )
        if args.freeze_initial_layers:
            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False
                if i == 34:
                    break


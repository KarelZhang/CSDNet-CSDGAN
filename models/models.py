
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'gan':
        from .gan_model import GanModel
        model = GanModel()
    elif opt.model == 'pair':
        # assert(opt.dataset_mode == 'unaligned')
        from .pair_model import PairModel
        model = PairModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

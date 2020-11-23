def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'registration_model_contrastive_learning':
        from .registration_model_contrastive_learning import RegistrationModel
        model = RegistrationModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model


cfg = {}

def model_configuration(args):
    if args.model == 'EncDec-AD':
        cfg['type'] = 'reconstruct'
        cfg['input_dim'] = args.n_sensor
        cfg['hidden_dim'] = 64
        cfg['num_layers'] = 2
        cfg['teacher_forcing_ratio'] = 0.5

    elif args.model == 'Transformer':
        cfg['type'] = 'forecast'
        cfg['input_dim'] = args.n_sensor
        cfg['hidden_dim'] = 64
        cfg['num_layers'] = 2
        cfg['num_heads'] = 4

    return cfg
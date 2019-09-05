def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in 
        model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d.value for d in dims]) for dims in shapes]
    return shapes


class Nontrainable(object):
    
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.trainable_status = self.model.trainable
        self.model.trainable = False
        return self.model

    def __exit__(self, type, value, traceback):
        self.model.trainable = self.trainable_status

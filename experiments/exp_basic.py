import torch

class PipelineTemplate(object):
    """ Model template class.

    Specifies functions that should be implemented in derived model.
    """
    def __init__(self, args):
        self.args = args
        self.model = self._build_model().cuda()

    def _build_model(self):
        raise NotImplementedError()

    def _get_data(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
    
    def valid(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
    
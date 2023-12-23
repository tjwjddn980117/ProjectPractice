from easydict import EasyDict as edict
import numpy as np

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, 
    clobbering the options in b whenever they are also specified in a.

    Arguments:
        a (edict):
        b (edict): 
    """

    if type(a) is not edict:
        return

    for key, value in a.items():
        # a must specify keys that are in b
        if key not in b:
            raise KeyError('{} is not a valid config key'.format(key))
        
        # the types must match, too
        old_type = type(b[key])
        if old_type is not type(value):
            # if a's value isn't Numpy, we should change value to Numpy
            if isinstance(b[key], np.ndarray):
                value = np.array(value, dtype=b[key].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[key]),type(value), key))
            
        # recursively merge dicts
        if type(key) is edict:
            try:
                _merge_a_into_b(a[key], b[key])
            except:
                print('Error under config key: {}'.format(key))
                raise
        else:
            b[key] = value

def merge_dicts():
    a = edict({'foo': 3, 'bar': np.array([1, 2])})
    b = edict({'foo': 4, 'bar': np.array([3, 4]), 'hello': 5})
    _merge_a_into_b(a, b)
    print(b)  # 출력: {'foo': 3, 'bar': array([1, 2])}

merge_dicts()

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )

    def forward(self, z_code, c_code=None):
        in_code = z_code
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
import torch.nn as nn

from netModule.fixed_gru import fixed_gru
from netModule.fixed_rnn import fixed_rnn

class char_cnn_rnn(nn.Module):
    '''
    Char-CNN-RNN model, described in ``Learning Deep Representations of
    Fine-grained Visual Descriptions``.

    This implementation supports two distinct incarnations of the models for
    the birds and flowers datasets:
        * ``Learning Deep Representations of Fine-Grained Visual Descriptions
            (https://github.com/reedscot/cvpr2016)
        * ``Generative Adversarial Text-to-Image Synthesis``
            (https://github.com/reedscot/icml2016)

    Each incarnation contains slight variations on the model architecture, and
    the architecture also varies depending on the dataset used.

    Arguments:
        dataset (string): which dataset was used.
        model_type (string): which incarnation of the model to use.
    '''
    def __init__(self, dataset, model_type):
        super(char_cnn_rnn).__init__()
        assert dataset in ['birds', 'flowers'], 'dataset should be (birds|flowers)'
        assert model_type in ['cvpr', 'icml'], 'model_type should be (cvpr|icml)'

        if model_type == 'cvpr':
            rnn_dim = 256
            use_maxpool3 = True
            rnn = fixed_rnn
            rnn_num_steps = 8
        else: # model_type == 'icml'
            rnn_dim = 512
            if dataset == 'flowers':
                use_maxpool3 = True
                rnn = fixed_rnn
                rnn_num_steps=8
            else: # dataset == 'birds'
                use_maxpool3 = False
                rnn = fixed_gru
                rnn_num_steps=18

        self.dataset = dataset
        self.model_type = model_type
        self.use_maxpool3 = use_maxpool3

        # network_setup
        # (B, 70, 201) , its (Batch, number_of_char, max_length)
        # (batch_size, in_channels, size) -> (batch_size, out_channels, resize)
        # if Conv1d (batch_size, in_channels, seq) -> (batch_size, out_channels, re_seq)
        # if Conv2d (batch_size, in_channels, (H*W)) -> (batch_size, out_channels, re(H*W))
        self.conv1 = nn.Conv1d(70, 384, kernel_size=4)
        self.threshold1 = nn.Threshold(1e-6, 0)
        # maxpoolint 1st dimension
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        # (B, 384, 66)
        self.conv2 = nn.Conv1d(384, 512, kernel_size=4)
        self.threshold2 = nn.Threshold(1e-6, 0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        # (B, 512, 21)
        self.conv3 = nn.Conv1d(512, rnn_dim, kernel_size=4)
        # (B, rnn_dim, 18) 
        self.threshold3 = nn.Threshold(1e-6, 0)

        if use_maxpool3:
            # make (B, rnn_dim, 8)
            self.maxpool3 = nn.MaxPool1d(kernel_size=3,stride=2)
        
        # (B, rnn_dim, rnn_num_steps)
        self.rnn = rnn(num_steps=rnn_num_steps, emb_dim=rnn_dim)

        # (B, rnn_dim)
        self.emb_proj = nn.Linear(rnn_dim, 1024)
        # (B, 1024)

    def forward(self, x):
        # temporal convolutions
        out = self.conv1(x)
        out = self.threshold1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.threshold2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.threshold3(out)
        if self.use_maxpool3:
            out = self.maxpool3(out)
        
        # resize (B, rnn_dim, rnn_num_steps) to (B, rnn_num_steps, rnn_dim)
        out = out.permute(0, 2, 1)
        out = self.rnn(out)

        # linear projection
        out = self.emb_proj(out)

        return out
# -*-Encoding: utf-8 -*-

"""
Description: A GRU-based baseline model to forecast future wind power
Authors: Chen

"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

#LSTM model;
class BaselineLSTMModel(nn.Layer):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineLSTMModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.LSTM(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:  就是batch_x
        Returns:
            A tensor
        """
        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])   
        x_enc = paddle.concat((x_enc, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out:]
        return sample  # [B, L, D]  batch, output_length, features_dimsion

#RNN model;
class BaselineRNNModel(nn.Layer):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineRNNModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:  就是batch_x
        Returns:
            A tensor
        """
        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])   
        x_enc = paddle.concat((x_enc, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out:]
        return sample  # [B, L, D]  batch, output_length, features_dimsion


class En_Decoder_GRU(paddle.nn.Layer):
    def __init__(self, settings,use_teacher_forcing = True):
        super(En_Decoder_GRU, self).__init__()
        
        
        self.output_len = settings["output_len"]
        self.input_len = settings["input_len"]
        self.use_teacher_forcing = use_teacher_forcing
        self.out = settings["out_var"]
        self.hidC = settings["in_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        
        self.hidR = 48
        #encoder
        self.lstm1 = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"])   
        self.lstm2 = nn.GRU(input_size= 3 + self.hidR,  
                                   hidden_size=self.hidR, num_layers=1)
        
        #MLP
        self.projection = nn.Linear(self.hidR, self.out)
        
        
        # for computing output logits
        self.outlinear =paddle.nn.Linear(self.hidR, self.output_len)
        
    def forward(self, x_enc, xf, target):   
        """
        xf:future features
        previous hidden:
        """
        
        batch_size = xf.shape[0]
        enc_lstmout, _ = self.lstm1(x_enc)  #seq, batch, dims
        
        ##decoder
        # Creating first decoder_hidden_state = 0
        decoder_hidden = paddle.zeros(shape=[batch_size, 1, self.hidR], dtype="float32")  
        # Initializing predictions vector
        outputs = paddle.zeros(shape=[batch_size, self.output_len, 1], dtype="float32")

        # Initializing first prediction
        decoder_output = paddle.zeros(shape=[batch_size, 1, 1],dtype="float32")  

        # List of alphas, for attention check
        attn_list = []
        
        for t in range(self.output_len):
            
            context_vector_or = paddle.unsqueeze(paddle.sum(enc_lstmout, 1),1)
            x_input = paddle.concat((xf[:,t,:].unsqueeze(1), decoder_output, context_vector_or), axis=-1)   
            
            # GRU decoder
            
            decoder_hidden = paddle.transpose(decoder_hidden, [1,0,2])
            
            dec_lstmout, d_hidden = self.lstm2(x_input, decoder_hidden)  #
            
            decoder_output = self.projection(self.dropout(dec_lstmout))
            decoder_hidden = paddle.transpose(d_hidden, [1,0,2])
            #decoder_hidden = d_hidden  

            outputs[:,t,:] = decoder_output.squeeze(1)

        return outputs, attn_list
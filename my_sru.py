import torch
import math

class SRUCell(torch.nn.Module): 
    """naive implementation mimicking behavior of LSTMCell and GRUCell"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # parameters
        self.W_all = torch.nn.Parameter(torch.empty(input_size, 3 * hidden_size)) # unified: W / W_f / W_r
        self.v = torch.nn.Parameter(torch.empty(2 * hidden_size)) # unified: v_f / v_r
        self.b = torch.nn.Parameter(torch.empty(2 * hidden_size)) # unified: b_f / b_r

        # initialzation
        self.init_params()

    def init_params(self): 
        bound = math.sqrt(3.0 / self.hidden_size)
        torch.nn.init.uniform_(self.W_all, -bound, bound)
        torch.nn.init.uniform_(self.v, -bound, bound)
        torch.nn.init.zeros_(self.b)
    
    def forward(self, input, prev_hidden=None): 
        I, B, D = self.input_size, input.shape[0], self.hidden_size

        if prev_hidden is None: 
            prev_hidden = torch.zeros(B, D, device=input.device)
        taller_hidden = torch.cat((prev_hidden, prev_hidden), 1)

        U = input @ self.W_all # batched mul

        # shape-shift input for highway network connection
        if I < D: 
            input_shaped = torch.nn.functional.pad(input, (1, D-I)) # pad
        elif I > D: 
            input_shaped = input[:,:D] # truncate
        else: 
            input_shaped = input # keep

        fr = (U[:,D:] + self.v * taller_hidden + self.b).sigmoid() # f / r 2-in-1
        f, r = fr[:,:D], fr[:,D:]
        c = f * prev_hidden + (1 - f) * U[:,:D]
        h = r * c + (1 - f) * input_shaped * math.sqrt(3.0) # correction term with b=0

        return h, c

class SRUOneDirOnly(torch.nn.Module): 
    """simple implementation based on SRUCell (ONE DIR ONLY)"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)
        # one for each layer (first layer with different input size)
        self.sru_cells = torch.nn.ModuleList([SRUCell(input_size, hidden_size)]) # first layer
        self.sru_cells.extend([SRUCell(hidden_size, hidden_size) for _ in range(num_layers-1)])

    def forward(self, input, init_hidden=None): 
        # input shape: (L, B, I), init_hidden shape: (N, B, D)
        L, B, D, N = input.shape[0], input.shape[1], self.hidden_size, self.num_layers
        if init_hidden is None: 
            init_hidden = torch.zeros(N, B, D, device=input.device)
        
        h_for_return = torch.empty(L, B, D, device=input.device)
        c_for_return = torch.empty(N, B, D, device=input.device)

        for j in range(N): # per layer
            cur_input = input if (j == 0) else h_for_return # first layer special! 
            cur_sru_cell = self.sru_cells[j]
            h_now = torch.empty(L, B, D, device=input.device)
            c_now = init_hidden[j]
            for i in range(L): # per word batch
                h, c = cur_sru_cell(cur_input[i], c_now)
                h_now[i] = h
                c_now = c
            if j != N-1: 
                h_now = self.dropout(h_now)
            h_for_return = h_now
            c_for_return[j] = c_now

        return h_for_return, c_for_return

class SRU(torch.nn.Module): 
    """full SRU model, with bidirectional support"""
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.sru_forward = SRUOneDirOnly(self.input_size, self.hidden_size, self.num_layers, self.dropout)
        if self.bidirectional:
            self.sru_reverse = SRUOneDirOnly(self.input_size, self.hidden_size, self.num_layers, self.dropout)

    def forward(self, input, init_hidden=None): 
        if type(input) == torch.nn.utils.rnn.PackedSequence: # weird DrQA thing
            parsed_input, lengths = torch.nn.utils.rnn.pad_packed_sequence(input)
        else: 
            parsed_input = input
        
        if self.bidirectional: 
            forward_h, forward_c = self.sru_forward(parsed_input, init_hidden)
            reverse_h, reverse_c = self.sru_reverse(torch.flip(parsed_input, [0]), init_hidden)
            h = torch.cat((forward_h,reverse_h), 2)
            c = torch.cat((forward_c,reverse_c), 2)
        else: 
            h, c = self.sru_forward(parsed_input, init_hidden)
        
        if type(input) == torch.nn.utils.rnn.PackedSequence: # weird DrQA thing (again)
            h = torch.nn.utils.rnn.pack_padded_sequence(h, lengths)
        
        return h, c
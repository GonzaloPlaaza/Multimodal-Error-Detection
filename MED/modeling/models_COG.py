import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import clip

from .models_TCN import DilatedResidualLayer, ScaledDotProductAttention, PoswiseFeedForwardNet

"""
Chain-of-Gesture (COG)
"""

class MultiHeadAttention_COG(nn.Module):
    def __init__(self, d_model, d_q, n_heads):
        super(MultiHeadAttention_COG, self).__init__()

        self.W_Q = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_q, d_model, bias=False)

        self.d_model = d_model
        self.d_q = d_q
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_q, n_heads)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]

        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_q)  # context: [batch_size, len_q, n_heads * d_v]
        output = context  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to("mps")(output + residual), attn
    

class SingleStageModel1_COG(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 kernel_size,
                 causal_conv = True, dropout = False, hier = False, use_output = False):
        super(SingleStageModel1_COG, self).__init__()
        self.use_output = use_output
        if self.use_output:
            self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = dropout
        self.hier = hier
        if dropout:
            self.channel_dropout = nn.Dropout2d()
        if hier:
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x

        if self.dropout:
            out = out.unsqueeze(3)
            out = self.channel_dropout(out)
            out = out.squeeze(3)
        
        for layer in self.layers:
            out = layer(out)
        if self.hier:
            f = self.pool(out)
        else:
            f = out

        out_classes = self.conv_out_classes(f)
        return f, out_classes
    
class TransformerCOT(nn.Module):
    
    def __init__(self, d_model, d_ff, d_q, n_layers, n_heads, device):
        
        super(TransformerCOT, self).__init__()
        self.layer1 = Encoder_COG(d_model, d_ff, d_q, n_layers, n_heads).to(device) 
        self.atten = MultiHeadAttention_COG(d_model = d_model, d_q = d_model, n_heads=1)

    def forward(self, visual, text):
        '''
        This function takes in inputs of sizes:
            visual: [batch_size, len_q, d_model]
            text: [batch_size, num_gest, d_model]

        to return:
            dec_outputs: [batch_size, len_q, d_model]
        '''
    
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_cross_attns = self.layer1(visual, text) # [batch_size, src_len, d_model]
        dec_outputs, dec_attns = self.atten(enc_outputs, text, text) #[batch_size, src_len, d_model]
        
        return dec_outputs  
    

class MyTransformer(nn.Module):
    def __init__(self, f_dim, gest_f_dim, d_model, d_q, len_q, device):
        super(MyTransformer, self).__init__()
        
        self.dim = f_dim  # 2048
        self.dim2 = gest_f_dim # 512
        self.len_q = len_q #len_q is length of sequence in transformer; i.e., the number of frames looked at (t-n+1 to t)
        self.d_model = d_model #d_model is the dimension of the model and the scaling factor
        self.d_q = d_q #d_q is the dimension of the query
        self.device = device

        #The two linear layers are used to project the visual and text features into the same dimension
        self.linear1 = nn.Linear(f_dim, d_model, bias=False)
        self.linear2 = nn.Linear(gest_f_dim, d_model, bias=False)
        self.transformer = TransformerCOT(d_model = d_model, d_ff= f_dim,  d_q = d_q,
                                        n_heads = 8, n_layers = 2, device = device)

        #self.fc1 = nn.Linear(gest_f_dim, out_features, bias = False)
        #self.fc2 = nn.Linear(gest_f_dim, gest_features, bias = False)
        #self.linear = nn.Linear(gest_features*d_model, num_f_maps)


    def forward(self, g, long_feature):
        # g: gesture prompt [15, 768]
        # long_feature: visual feature [1, total_frame, 2048]
        visual = self.linear1(long_feature) # 1, 345, 2048 -> 1, 345, d_model
        text = self.linear2(g) #1, 15, 768 -> 1, 15, d_model

        inputs = []
        frame_length = visual.size(1)

        #For each frame, we look at the previous len_q frames
        for i in range(frame_length):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.d_model)).to(self.device)
                input = torch.cat([input, visual[:, 0:i+1]], dim=1)
            else:
                input = visual[:, i-self.len_q+1:i+1]
            inputs.append(input)
        
        visual_feas = torch.stack(inputs, dim=0).squeeze(1) # [total_frame, len_q, d_model] = [345, 30, d_model]
        text_feas = [text for _ in range(frame_length)]
        text_feas = torch.stack(text_feas, dim = 0).squeeze(1) # [total_frame, num_gest, num_dim] = [345, 15, d_model]
        
        output = self.transformer(visual_feas, text_feas) #[345,15,d_model]

        outputs = output.reshape(frame_length, -1) 
        #output1 = self.linear(outputs) #[345, num_f_maps]
        #output2 = self.fc22(outputs) #[345,15]
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return outputs.unsqueeze(0)#, output2.unsqueeze(0) [1, 345, num_gest * d_model]
    

class FPN(nn.Module):
    def __init__(self,num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d( num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = nn.Conv1d( num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,W = y.size()
        return F.interpolate(x, size=W, mode='linear') + y

    def forward(self,out_list):
        #p3 = out_list[2]
        #c2 = out_list[1]
        #c1 = out_list[0]
        #p2 = self._upsample_add(p3, self.latlayer1(c2))
        #p1 = self._upsample_add(p2, self.latlayer1(c1))
        #return [p1,p2,p3]
        
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1,p2,p3,p4]
    
class EncoderLayer_COG(nn.Module):
    def __init__(self, d_model, d_ff, d_q, n_heads):
        
        super(EncoderLayer_COG, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.enc_self_attn = MultiHeadAttention_COG(d_model, d_q, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, Q, K, V):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        Q = self.norm1(Q)
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(Q, K, V)  # Q, K, V are text, visual, visual
        enc_outputs = self.pos_ffn(self.norm3(enc_outputs))  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class Encoder_COG(nn.Module):
    
    def __init__(self, d_model, d_ff, d_k, n_layers, n_heads):
        
        super(Encoder_COG, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([EncoderLayer_COG(d_model, d_ff, d_k, n_heads) for _ in range(n_layers)]) #n_layers = 2

    def forward(self, visual, text):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        visual = self.norm(visual)
        for layer in self.layers:
            text, attn = layer(text, visual, visual)

        return text, attn
    

class COG(nn.Module):
    def __init__(self, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_template = 'A surgeon is', gest_prompt: str = './data/prompts/gest_prompt.pt'):
        super(COG, self).__init__()
        
        self.name = 'COG'
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512 # 768
        
        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu') #'ViT-L/14'
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) #(1, 512)

        for i in range(num_gest):
            # Encode each of the gestures using CLIP
            gest_prompt = text_model.encode_text(clip.tokenize(f'{gest_template} {self.gest_list[i]} ...')).float()
            self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
        
        self.all_gest_fea = self.all_gest_fea[1:]
        torch.save(self.all_gest_fea, self.gest_prompt) #self.gest_prompt is the path to save the gesture embeddings
        
        self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        
        #else:
         #   self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        ##slow path
        self.TCN = SingleStageModel1_COG(num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1_COG(num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=16, stride=16)
        self.fast_stage1 = SingleStageModel1_COG(num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1_COG(num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    
    def forward(self, x):
        # x: visual feature [1, 345, 2048]
        
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]

        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1) #[1, 345, num_gest * d_model] -> [1, num_gest * d_model, 345]
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list

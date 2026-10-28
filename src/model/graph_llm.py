import contextlib
import torch
import time
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch_scatter import scatter
from src.model.gnn import load_gnn_model
from timm.models.layers import drop_path
from timm.layers import Mlp
import copy
from itertools import chain
import math

import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100



TEMP_MIN = 0.5
ANNEAL_RATE = 0.00003
EPS = 1e-20




class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x.to(self.norm1.weight.dtype))))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x




class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '11GiB', 1: '11GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, 
                                                        use_fast=False, 
                                                        revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'


        model = AutoModelForCausalLM.from_pretrained(
                args.llm_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                **kwargs)



        if args.llm_frozen == 'True':
            print("Freezing!")
            print(args.llm_model_path)
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            #lora_r: int = 8
            lora_r: int = 128
            lora_alpha: int = 16
            lora_dropout: float = 0.05

            lora_target_modules = [
                    "q_proj",
                    "v_proj",
                ]

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                #layers_to_transform=layers_to_transform,
            )
            model = get_peft_model(model, config)
            

        self.model = model
        print('Finish loading')
        print(args.llm_model_path)


        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers//2,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.graph_decoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_hidden_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers//2,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        


        self.mask_ratio = args.mask_ratio
        self.llm_name = args.llm_model_path
        self.hidden_dim = args.gnn_hidden_dim
        
        if 'Llama-3.1-8B' in args.llm_model_path:
            self.llm_dim = 4096
        if '7b'  in args.llm_model_path:
            self.llm_dim = 4096
        if 'Llama-3.2-3B' in args.llm_model_path:
            self.llm_dim = 3072
        if 'Llama-3.2-1B' in args.llm_model_path:
            self.llm_dim = 2048
        if 'Qwen3.5-2B' in args.llm_model_path:
            self.llm_dim = 2048
        if 'Qwen3.5-0.8B' in args.llm_model_path:
            self.llm_dim = 1024


        self.encoder = nn.ModuleList([Block(dim=args.gnn_hidden_dim, num_heads=8, mlp_ratio=args.mlp_ratio, 
                                      qkv_bias=False, qk_scale=None,drop=0., attn_drop=0.00, 
                                      drop_path=0.00, norm_layer=nn.LayerNorm,init_values=0.)
        for i in range(args.encoder_depth)]).to(self.model.device)

        self.decoder = nn.ModuleList([Block(dim=args.gnn_hidden_dim, num_heads=8, mlp_ratio=args.mlp_ratio, 
                                      qkv_bias=False, qk_scale=None,drop=0., attn_drop=0.00, 
                                      drop_path=0.00, norm_layer=nn.LayerNorm,init_values=0.)
        for i in range(args.decoder_depth)]).to(self.model.device)

        self.get_token_probs = nn.Sequential(
                                Block(dim=args.gnn_hidden_dim, num_heads=8, mlp_ratio=args.mlp_ratio, 
                                      qkv_bias=False, qk_scale=None,drop=0.01, attn_drop=0.00, 
                                      drop_path=0.00, norm_layer=nn.LayerNorm,init_values=0.),
                                nn.Linear(args.gnn_hidden_dim, 1),
                                torch.nn.Flatten(start_dim=1),
                                ).to(self.model.device)

        self.sampling_pos = nn.Conv1d(in_channels=args.gnn_hidden_dim, 
                                        out_channels=args.gnn_hidden_dim, 
                                        kernel_size=1, 
                                        stride=1,
                                        groups = args.gnn_hidden_dim).to(self.model.device)
        
        self.norm_e =  nn.LayerNorm(args.gnn_hidden_dim).to(self.model.device)
        self.norm_d =  nn.LayerNorm(args.gnn_hidden_dim).to(self.model.device)
        self.head_e = nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim).to(self.model.device)
        self.head_d = Mlp(in_features=args.gnn_hidden_dim, hidden_features=args.gnn_hidden_dim*4, act_layer=nn.GELU, drop=0.).to(self.model.device)
        self.encoder2decoder = nn.Linear(args.gnn_hidden_dim, args.gnn_hidden_dim).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, self.llm_dim//2),
            nn.Sigmoid(),
            nn.Linear(self.llm_dim//2, self.llm_dim),
        ).to(self.model.device)


        self.softmax =  nn.Softmax(dim=-1)
        self.word_embedding = self.model.model.get_input_embeddings()


        self.loss_func = nn.MSELoss(reduction="none")
        #self.loss_func = nn.SmoothL1Loss(reduction="none", beta=2.0)

        self.loss_type = args.loss_type
        self.dataset = args.dataset


        self.mask_token_pos = nn.Conv1d(in_channels=args.gnn_hidden_dim, 
                                        out_channels=args.gnn_hidden_dim, 
                                        kernel_size=1, 
                                        stride=1,
                                        groups = args.gnn_hidden_dim).to(self.model.device)
        self.vis_token_pos = nn.Conv1d(in_channels=args.gnn_hidden_dim, 
                                        out_channels=args.gnn_hidden_dim, 
                                        kernel_size=1, 
                                        stride=1,
                                        groups = args.gnn_hidden_dim).to(self.model.device)
        self.encoder_pos = nn.Conv1d(in_channels=args.gnn_hidden_dim, 
                                        out_channels=args.gnn_hidden_dim, 
                                        kernel_size=1, 
                                        stride=1,
                                        groups = args.gnn_hidden_dim).to(self.model.device)

        self.old_policy_probs = None  # Initialize storage for old_p_x


    #'''  EMA 
        self.initialize_ema_encoder()

    def initialize_ema_encoder(self):
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.target_pos = copy.deepcopy(self.encoder_pos)
        self.target_norm = copy.deepcopy(self.norm_e)

        self.update_ema_components(initial=True)
        self.ema_decay = 0.999

    def update_ema_components(self, initial=False):
        if initial:
            for ema_block, primary_block in zip(self.ema_encoder, self.encoder):
                ema_block.load_state_dict(primary_block.state_dict())

            self.target_norm.load_state_dict(self.norm_e.state_dict())

            for ema_param, primary_param in zip(self.target_pos.parameters(), self.encoder_pos.parameters()):
                ema_param.data.copy_(primary_param.data)
            print("EMA initial")

        else:
            for ema_block, primary_block in zip(self.ema_encoder, self.encoder):
                for ema_param, primary_param in zip(ema_block.parameters(), primary_block.parameters()):
                    ema_param.data.mul_(self.ema_decay).add_(primary_param.data * (1.0 - self.ema_decay))
            
            for ema_param, primary_param in zip(self.target_pos.parameters(), self.encoder_pos.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(primary_param.data * (1.0 - self.ema_decay))
            
            for ema_param, primary_param in zip(self.target_norm.parameters(), self.norm_e.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(primary_param.data * (1.0 - self.ema_decay))
            print("EMA update") 

    def forward_target(self, n_embeds): # EMA Target
        with torch.no_grad():
            x_label = n_embeds + self.target_pos(n_embeds.transpose(1, 2)).transpose(1, 2).clone().detach()
            for encoder in self.ema_encoder:
                x_label = encoder(x_label)
            x_label = self.target_norm(x_label)
            return x_label
    #''' 


    #'''
    #def forward_target(self, n_embeds): # Simple Target
    #    with torch.no_grad():
    #        x_label = n_embeds + self.encoder_pos(n_embeds.transpose(1,2)).transpose(1,2).clone().detach()
    #        for encoder in self.encoder:
    #            x_label = encoder(x_label)
    #        x_label = self.norm_e(x_label)
    #        return x_label
    #'''


    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def loss_forward(self, x, ema_x):
        loss = self.loss_func(input=x, target=ema_x.detach())  # no grad to ema_x
        return loss

    def get_mask(self, x):
        x = x + self.sampling_pos(x.transpose(1,2)).transpose(1,2).clone()
        logits = self.get_token_probs(x)
        logits =  torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        visible_tokens = int(logits.shape[1] * (1 - self.mask_ratio))

        vis_idx = torch.multinomial(p_x, num_samples=visible_tokens, replacement=False)
        mask = torch.ones((x.shape[0], x.shape[1])).to(x.device, non_blocking=True)
        mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0)
        mask = mask.flatten(1).to(torch.bool)
        return p_x, vis_idx, mask

    def forward_target(self, n_embeds):
        with torch.no_grad():
            x_label = n_embeds + self.encoder_pos(n_embeds.transpose(1,2)).transpose(1,2).clone().detach()
            for encoder in self.encoder:
                x_label = encoder(x_label)
            x_label = self.norm_e(x_label)
            return x_label

    def encode_graphs(self, samples,train_stage):
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)

    ### Graph Encoder ###
        n_embeds, edge_attr = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
        n_embeds = n_embeds.unsqueeze(0)

        B, N, C = n_embeds.shape

    ### Target Encoder
        x_label = self.forward_target(n_embeds)   # Apply Target Encoder 
        #x_label = n_embeds                       # Target Encoder Free

    ### Masking guided ###
        p_x, _, mask = self.get_mask(n_embeds)
        x_vis = n_embeds[~mask].reshape(B, -1, C)
        x_mask = n_embeds[mask].reshape(B, -1, C) 
        #print(x_mask)

    ### Mask Encoder ###
        x_vis = x_vis + self.encoder_pos(x_vis.transpose(1,2)).transpose(1,2).clone().detach()
        for encoder in self.encoder:
            x_vis = encoder(x_vis)
        x_vis = self.head_e(self.norm_e(x_vis))

    ### Latent ###
        x_vis = self.encoder2decoder(x_vis)  # Vis
        x_mask = x_mask + self.mask_token_pos(x_mask.transpose(1,2)).transpose(1,2).clone().detach()
        x_vis = x_vis + self.vis_token_pos(x_vis.transpose(1,2)).transpose(1,2).clone().detach()
        n_embeds_out = torch.cat([x_vis, x_mask], dim=1)

    ### Mask Decoder ###
        for decoder in self.decoder:
            n_embeds_out = decoder(n_embeds_out)

        n_embeds_predit=self.norm_d(n_embeds_out)
        x_vis = n_embeds_predit[:, :x_vis.shape[1]]
        x_mask = self.head_d(n_embeds_predit[:, -x_mask.shape[1]:])

    ### GraphDecoder
        n_embeds, edge_attr = self.graph_decoder(n_embeds_out.squeeze(0), graphs.edge_index.long(), edge_attr)

    # Mean pooling
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        
        return g_embeds, x_label, x_mask, x_vis, p_x, mask


    def forward(self, samples):
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        train_stage = True
        g_embeds, graph_l, mask_outputs, vis_outputs, p_x, bool_masked_pos = self.encode_graphs(samples,train_stage)
        graph_embeds = self.projector(g_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)


        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)



        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        
        ############################################ Graph Masking ###################################################
        B, _, C = graph_l.shape
        mask_labels = graph_l[bool_masked_pos].reshape(B, -1, C)
        
        if mask_outputs.shape==mask_labels.shape:
            if 'mse' in self.loss_type:
                mask_l_r = torch.mean(self.loss_forward(mask_outputs, mask_labels.detach()),dim=-1)


            l_s = torch.zeros(graph_l.shape[0], ).to(mask_l_r.device)
            for i in range(p_x.shape[0]):
                    m = torch.distributions.categorical.Categorical(probs=p_x[i])
                    log_probs = m.log_prob(torch.arange(0, p_x.shape[1], 1).to(p_x.device)) 
                    mask_log_probs = log_probs[bool_masked_pos[i]]
                    reward_g = mask_l_r[i].detach() #* reward[i].detach()
                    l_s[i] = -torch.mean(mask_log_probs*reward_g)
            m_l_r = torch.mean(mask_l_r)         # Mask Reconstruction loss
            m_l_s = 1e-4 * torch.mean(l_s)                       # Sampling 
            #rf_loss = (1.3 * m_l_r) + (0.7 *m_l_s)              # Total loss
            rf_loss = m_l_r + m_l_s               # Total loss
            outputs.loss = outputs.loss * 1

            
    ########################################################################################################################
        return outputs.loss, rf_loss






    def inference(self, samples):
        start_time = time.time()  # Start time

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        train_stage = False
        g_embeds, _, _, _, _, _ = self.encode_graphs(samples,train_stage)

        graph_embeds = self.projector(g_embeds)


        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        total_tokens = 0  # Initialize total tokens counter

        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids

            total_tokens += len(input_ids)  # Count tokens

            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(pred)

        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        tokens_per_second = total_tokens / elapsed_time  # Calculate tokens per second

        print(f"Tokens per second: {tokens_per_second}")



        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }



    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

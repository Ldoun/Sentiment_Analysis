from .model import *

def args_for_model(parser, model):
    parser.add_argument('--num_classes', type=int, default=3, help='num_classes')
    parser.add_argument('--input_size', type=int, default=256, help='emb size')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=512)
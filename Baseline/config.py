model_name = "bert-base-uncased"
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
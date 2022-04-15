class SimCSECollator(object):
    """SimCSE Collator"""
    def __init__(self,
                 tokenizer,
                 features = ("input_ids", "attention_mask", "token_type_ids"),
                 max_len = 100):
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len

    def collate(self, batch):
        new_batch = []
        for example in batch:
            for i in range(2):
                # repeat twice
                new_batch.append({fea: example[fea] for fea in self.features})
        new_batch = self.tokenizer.pad(
            new_batch,
            padding = True,
            max_length = self.max_len,
            return_tensors = "pt"
        )
        return new_batch

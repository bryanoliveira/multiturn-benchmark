from transformers import StoppingCriteria

class StrStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_length, stop_str):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_str = stop_str

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_str in self.tokenizer.decode(input_ids[0][self.start_length :], skip_special_tokens=True)
import torch
import torch.nn.functional as F

class CTCBeamSearchDecoder(torch.nn.Module):
    def __init__(self, vocabulary, beam_width=3, blank_id=0):
        super().__init__()
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.vocabulary = vocabulary

    def forward(self, logits):
        batch_size, seq_len, num_classes = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        decoded_batch = []
        for b in range(batch_size):
            beams = [([], 0)]  # (prefix, accumulated_prob)

            for t in range(seq_len):
                new_beams = []
                for prefix, accumulated_prob in beams:
                    for c in range(num_classes):
                        prob = probs[b, t, c]
                        log_prob = torch.log(prob)

                        if c == self.blank_id:
                            new_prefix, new_prob = prefix, accumulated_prob + log_prob
                        else:
                            new_prefix, new_prob = prefix + [c], accumulated_prob + log_prob

                        new_beams.append((new_prefix, new_prob))

                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:self.beam_width]

            best_beam = beams[0][0]
            decoded_batch.append(self.postprocess(best_beam))

        return decoded_batch

    def postprocess(self, beam):
        decoded_chars = []
        for i, char_id in enumerate(beam):
            if i > 0 and char_id == beam[i-1]:
                continue
            if char_id != self.blank_id:
                # print(char_id)
                decoded_chars.append(self.vocabulary[char_id])
        return ''.join(decoded_chars)
import torch


class TripletCollator:
    """
    Collator for triplet data with semi-hard and hard negative samples
    """

    def __call__(self, batch):
        batch_size = len(batch)
        num_contents = len(batch[0]['contents'])

        # query: a list of dicts from tokenizer
        queries = [item['query'] for item in batch]
        pos_masks = [item['pos_mask'] for item in batch]

        # Tokenize queries
        query_tokens = {
            k: torch.stack([q[k] for q in queries], dim=0)  # (batch_size, seq_len)
            for k in queries[0]
        }

        # Flatten all content dicts
        flattened_contents = [content for item in batch for content in item['contents']]

        # Tokenize contents (rebuild batched tensor)
        content_tokens = {
            k: torch.stack([c[k] for c in flattened_contents], dim=0)
            .view(batch_size, num_contents, -1)
            for k in flattened_contents[0]
        }

        return {
            'query': query_tokens,
            'contents': content_tokens,
            'pos_mask': torch.tensor(pos_masks, dtype=torch.bool)
        }

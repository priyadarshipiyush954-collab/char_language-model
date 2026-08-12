

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
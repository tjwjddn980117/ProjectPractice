SKIPGRAM_N_WORDS = 2

batch_input, batch_output = [], []
batch = [[1,2,3,4,5,6,7],
         [10,20,30,40,50,60,70],
         [100,200,300,400,500,600,700]]

for text in batch:
    # this 'text_pipeline' is lambda x: vocab(tokenizer(x))
    # the function work that 'text to index'
    # this can possible because vocab is type of dictionary. (let's check it. that's not a function)
    for idx in range(len(text) - SKIPGRAM_N_WORDS * 2):
        token_id_sequence = text[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
        # input wil int
        # outputs will [1,2,4,5]
        input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
        outputs = token_id_sequence
        for output in outputs:
            # batch_input will [3,3,3,3,4,4,4,4,5,5,5,5]
            # batch_output will [1,2,4,5,2,3,5,6,3,4,6,7]
            batch_input.append(input_)
            batch_output.append(output)


print(batch_input)
print(batch_output)

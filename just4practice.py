from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# 토크나이저 설정
tokenizer = get_tokenizer('basic_english')

# 데이터 반복자 생성
data_iter = ["This is an example sentence", "This is another sentence", "This is is is"]

# 각 문장을 토크나이즈한 후, 이를 하나의 반복자로 만듭니다.
tokenized_iter = map(tokenizer, data_iter)

# build_vocab_from_iterator 함수를 사용하여 어휘 사전 구축
vocab = build_vocab_from_iterator(tokenized_iter, specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])

# 어휘 사전 출력
print(vocab)
print(type(vocab))
print(vocab.get_itos())
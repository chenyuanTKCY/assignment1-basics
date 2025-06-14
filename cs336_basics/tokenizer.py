import pickle
import regex as re
from typing import Iterable, Iterator
import json
from .tokenizer_word import Word, PAT

class Tokenizer:

    def __init__(self,vocab, merges, special_tokens=None):
        # 初始化分词器
        # vocab: 词汇表字典，键为int类型id，值为bytes类型token
        # merges: BPE合并规则列表
        # special_tokens: 特殊token列表
        self.vocab: dict[int, bytes] = vocab  # id到token的映射
        self.mapping: dict[bytes,int] = {v:k for k, v in vocab.items()}  # token到id的反向映射
        self.merges: dict[tuple[bytes,bytes],int]={
            merge: idx for idx, merge in enumerate(merges)
        }  # 合并规则及其优先级索引
        self.special_tokens: list[str] | None = special_tokens  # 特殊token列表

        # 将特殊token添加到词汇表中
        if special_tokens:
            for special_token in self.special_tokens:
                special_token = special_token.encode("utf-8")
                if special_token not in self.mapping:
                    idx = len(self.vocab)
                    self.vocab[idx] = special_token
                    self.mapping[special_token] = idx

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从vocab和merges文件创建分词器实例"""
        # 加载vocab文件(JSON格式)
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        vocab = {v: k.encode("utf-8") for k, v in vocab.items()}

        # 加载merges文件(每行包含两个待合并的token)
        merges = []
        with open(merges_filepath) as m:
            for line in m:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(
                        tuple([b.encode("utf-8") for b in cleaned_line.split(" ")])
                    )
        return Tokenizer(vocab,merges,special_tokens)

    @classmethod
    def from_pickles(cls, pickle_filepath, special_tokens=None):
        """从pickle文件加载分词器实例"""
        with open(pickle_filepath,"rb") as f:
            data = pickle.load(f)
            return Tokenizer(data["vocab"],data["merges"],special_tokens)

    def encode(self,text:str)-> list[int]:
        """将文本编码为token id列表"""
        res = []

        # 处理特殊token分割
        if self.special_tokens:
            # 构建特殊token的正则匹配模式
            resplit = "|".join(
                [
                    "("+re.escape(s)+")"
                    for s in sorted(self.special_tokens, key=len, reverse=True)
                ]
            )
            docs = re.split(resplit,text)
        else:
            docs = [text]

        # 处理每个文本片段
        for doc in docs:
            if not doc:
                continue
            # 如果是特殊token则直接映射
            if self.special_tokens and doc in self.special_tokens:
                res.append(self.mapping[doc.encode("utf-8")])
                continue

            # 使用正则匹配单词并应用BPE算法
            for word in re.finditer(PAT,doc):
                if word:
                    word: str = word.group(0).replace("\r","")
                    word = Word(word.encode("utf-8"))
                    if len(word)>1:
                        # 获取所有可合并的token对
                        pairs = [p for p in word.pairs() if p in self.merges]
                        # 执行BPE合并
                        while len(word.tokens) > 1 and len(pairs) > 0:
                            best = min(pairs, key=lambda p: self.merges[p])  # 选择优先级最高的合并对
                            word.merge(best, b"".join(best))
                            pairs = [p for p in word.pairs() if p in self.merges]
                    # 将token映射为id
                    res.extend([self.mapping[token] for token in word.tokens])
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码文本迭代器"""
        for part in self._load_until_special(iterable):
            for encoded in self.encode(part):
                yield encoded

    def _load_until_special(self, iterable: Iterable[str]) -> Iterator[str]:
        """
        按特殊token分割文本流
        生成以特殊token结尾的文本块
        """
        buffer: str = ""
        for chunk in iterable:
            buffer += chunk
            if self.special_tokens:
                for token in self.special_tokens:
                    if buffer.endswith(token):
                        yield buffer
                        buffer = ""
                        break
        if buffer:
            yield buffer

    def decode(self, ids: list[int]) -> str:
        """将token id列表解码为文本"""
        return (
            b"".join([
                self.vocab[token] for token in ids
            ])
        ).decode("utf-8",errors="replace")  # 使用错误替换策略解码
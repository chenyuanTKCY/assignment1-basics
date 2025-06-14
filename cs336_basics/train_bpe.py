import regex as re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries
from .tokenizer_word import Word, PAT

# 特殊标记
SPECIAL_TOKEN = "<|endoftext|>"

def pre_tokenize_async(
    text: str,
    special_tokens: list[str]
) -> tuple[list[tuple[str]], list[tuple[str]], dict[bytes, list[Word]]]:
    """异步预处理文本，统计词对、单词和词对到单词的映射"""
    pairs: dict[bytes, int] = defaultdict(int)  # 词对计数
    words: dict[Word, int] = defaultdict(int)  # 单词计数
    pair2words: dict[bytes, set[Word]] = defaultdict(set)  # 词对到单词的映射

    # 按特殊标记分割文本
    docs = re.split("|".join([re.escape(s) for s in special_tokens]), text)
    for doc in docs:
        if doc.strip():  # 忽略空部分
            for word in re.finditer(PAT, doc):  # 使用正则匹配单词
                if word:
                    word = word.group(0).replace("\r", "")
                    if not word:
                        continue
                    word = Word(word.encode("utf-8"))  # 转换为Word对象
                    words[word] += 1
                    word_pairs = word.pairs()  # 获取单词的所有词对
                    for pair in word_pairs:
                        pairs[pair] += 1
                        pair2words[pair].add(word)
    return (pairs, words, pair2words)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
    batch_size: int = 1,
    debug: bool = False) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """训练BPE分词器的主函数"""
    if debug:
        begin = time.time()

    # 初始化词汇表，先添加特殊标记
    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    l = len(vocab)

    # 添加基础字节到词汇表
    vocab |= {i+l:bytes([i]) for i in range(256)}
    merges = []  # 存储合并操作

    # 初始化统计数据结构
    pairs: dict[bytes,int] = defaultdict(int)
    words: dict[Word,int] = defaultdict(int)
    pair2words: dict[bytes,set[Word]] = defaultdict(set)

    def pre_tokenize(text: str) ->None:
        """单线程预处理函数"""
        docs = re.split("|".join([re.escape(s) for s in special_tokens]),text)
        for doc in docs:
            if doc.strip():
                for word in re.finditer(PAT,doc):
                    if word:
                        word = word.group(0).replace("\r","")
                        if not word:
                            continue
                        word = Word(word.encode("utf-8"))
                        words[word] +=1
                        word_pairs = word.pairs()
                        for pair in word_pairs:
                            pairs[pair] += 1
                            pair2words[pair].add(word)

    # 多进程处理
    if num_processes > 1:
        with open(input_path, "rb") as f:
            # 查找文件分块边界
            boundaries = find_chunk_boundaries(
                f, batch_size * num_processes, SPECIAL_TOKEN.encode("utf-8"))
            boundaries = list(zip(boundaries[:-1], boundaries[1:]))
            for i in range(batch_size):
                if debug:
                    print(f"Running batch {i + 1} of {batch_size}")
                batch = i * num_processes
                with Pool(num_processes) as pool:
                    results = []
                    for start, end in boundaries[batch:batch + num_processes]:
                        f.seek(start)
                        chunk = f.read(end - start).decode("utf-8", errors="ignore")
                        if debug:
                            print(f"Processing chunk size {sys.getsizeof(chunk)} bytes")
                        # 异步处理每个分块
                        result = pool.apply_async(
                            pre_tokenize_async,
                            (chunk, special_tokens)
                        )
                        results.append(result)
                    pool.close()
                    # 合并结果
                    for result in results:
                        p, w, pw = result.get()
                        for pair, count in p.items():
                            pairs[pair] += count
                        for word, count in w.items():
                            words[word] += count
                        for pair, word_set in pw.items():
                            pair2words[pair].update(word_set)
                if debug:
                    print(f"Completed batch {i + 1} of {batch_size}")
    else:
        # 单线程处理
        with open(input_path,"rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, SPECIAL_TOKEN.encode("utf-8")
            )
            for start, end in zip(boundaries[:-1],boundaries[1:]):
                f.seek(start)
                chunk = f.read(end-start).decode("utf-8",errors="ignore")
                pre_tokenize(chunk)
    if debug:
        print(f"Pre-tokenization took {time.time()-begin:.2f} seconds")
        begin = time.time()

    # BPE合并过程
    while len(vocab)< vocab_size:
        # 找出频率最高的词对
        pair, _ = max(pairs.items(), key=lambda x:(x[1],x[0]))
        merged = b"".join(pair)
        merges.append((pair[0],pair[1]))  # 记录合并操作
        vocab[len(vocab)] = merged  # 添加到词汇表

        # 更新相关统计信息
        for word in pair2words[pair]:
            count = words[word]
            old_pairs = [p for p in word.pairs() if p != pair]
            for old_pair in old_pairs:
                pairs[old_pair] -= count
            word.merge(pair,merged)  # 合并词对
            new_pairs = [p for p in word.pairs() if p != pair]
            for new_pair in new_pairs:
                pairs[new_pair] += count
                pair2words[new_pair].add(word)
        del pairs[pair]
        del pair2words[pair]

    if debug:
        print(f"Merge took {time.time() - begin:.2f} seconds")

    return (vocab,merges)  # 返回词汇表和合并记录
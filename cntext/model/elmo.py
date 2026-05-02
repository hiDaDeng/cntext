import os
import gc
import torch
import hashlib
from typing import List, Optional, Dict
from pathlib import Path
from allennlp.modules.elmo import Elmo, batch_to_ids

class ELMoEmbedding:
    def __init__(self, 
                 model_dir: str = "./output/elmo_model",
                 emb_dir: str = "./output/elmo_emb",
                 batch_size: int = 64,
                 pre_trained: bool = True,
                 trainable: bool = False):
        """
        参数:
            model_dir: 模型缓存目录
            emb_dir: 词向量缓存目录
            batch_size: 批处理大小
            pre_trained: 是否使用预训练模型
            trainable: 是否微调模型
        """
        self.device = self._get_device()
        self.batch_size = batch_size
        self.pre_trained = pre_trained
        self.trainable = trainable
        
        # 初始化目录结构
        self.model_dir = Path(model_dir)
        self.emb_dir = Path(emb_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.model = self._load_model()
        
    def _get_device(self):
        """自动选择最佳计算设备"""
        return torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')

    def _load_model(self):
        """加载模型并处理缓存"""
        model_file = self.model_dir / "model.pt"
        
        if model_file.exists():
            return torch.load(model_file).to(self.device)
            
        if self.pre_trained:
            model = Elmo(
                options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                num_output_representations=1,
                requires_grad=self.trainable
            ).to(self.device)
        else:
            raise ValueError("自定义模型需要实现模型加载逻辑")
            
        torch.save(model, model_file)
        return model

    def _process_batch(self, batch: List[List[str]]) -> torch.Tensor:
        """处理单个批次并释放内存"""
        with torch.no_grad():
            char_ids = batch_to_ids(batch).to(self.device)
            embeddings = self.model(char_ids)['elmo_representations'][0]
            result = embeddings.cpu()
            del char_ids, embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            return result

    def get_embeddings(self, sentences: List[List[str]]) -> torch.Tensor:
        """
        获取句子嵌入，支持大语料处理
        
        参数:
            sentences: 已分词的句子列表
            
        返回:
            拼接后的嵌入张量 (num_sentences, max_len, embedding_dim)
        """
        # 检查缓存
        cache_key = hashlib.md5(str(sentences).encode()).hexdigest()
        cache_file = self.emb_dir / f"{cache_key}.pt"
        
        if cache_file.exists():
            return torch.load(cache_file).to(self.device)
            
        # 分批处理大语料
        results = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_emb = self._process_batch(batch)
            results.append(batch_emb)
            
        # 合并结果并保存缓存
        embeddings = torch.cat(results, dim=0)
        torch.save(embeddings, cache_file)
        
        return embeddings.to(self.device)

    @property
    def dim(self) -> int:
        """获取嵌入维度"""
        return self.model.get_output_dim()

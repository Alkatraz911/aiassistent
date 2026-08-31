"""Офлайн-диаризация для режима «один общий микрофон» (без токенов, без pyannote).

ECAPA-TDNN эмбеддинги (speechbrain, публичная модель `spkrec-ecapa-voxceleb`) +
агломеративная кластеризация по косинусу. Работает на уровне сегментов: один
эмбеддинг на реплику -> кластер -> метка спикера. Перекрывающаяся речь не
разводится (ограничение всех простых методов) — это fallback с ручной правкой.

Число голосов: если задано (интервьюер указал) — кластеризуем ровно в N; иначе
оцениваем по порогу расстояния.
"""
from __future__ import annotations

import numpy as np

from .. import config, device


class Diarizer:
    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"

    def _load(self) -> None:
        if self.model is not None:
            return
        import shutil
        from pathlib import Path

        import speechbrain.utils.fetching as sb_fetch
        from speechbrain.inference.speaker import EncoderClassifier

        # Windows без админ-прав не умеет symlink (WinError 1314). speechbrain тянет
        # файлы модели через link_with_strategy с symlink по умолчанию и не даёт
        # переопределить стратегию для всех файлов — поэтому форсируем копирование.
        def _copy_only(src, dst, *args, **kwargs):
            src, dst = Path(src), Path(dst)
            if dst.exists() or dst.is_symlink():
                try:
                    dst.unlink()
                except OSError:
                    pass
            shutil.copy(str(src), str(dst))
            return dst

        sb_fetch.link_with_strategy = _copy_only

        self.device = device.resolve_torch_device(config.FINALIZE_DEVICE)
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(config.BASE_DIR / "models" / "ecapa"),
            run_opts={"device": self.device},
        )
        import torch
        self._torch = torch

    def warmup(self) -> None:
        try:
            self._load()
        except Exception:
            pass

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """ECAPA-эмбеддинг куска аудио (float32 16кГц). None — если слишком коротко."""
        if audio.size < config.SAMPLE_RATE // 3:   # <~0.33 c — ненадёжно
            return None
        self._load()
        torch = self._torch
        with torch.inference_mode():
            wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            emb = self.model.encode_batch(wav).squeeze().cpu().numpy()
        return emb.astype(np.float32)

    def cluster(self, embeddings: list[np.ndarray], num_speakers: int | None) -> list[int]:
        """Кластеризует эмбеддинги -> метки спикеров (0..k-1)."""
        if len(embeddings) <= 1:
            return [0] * len(embeddings)
        from sklearn.cluster import AgglomerativeClustering

        X = np.vstack(embeddings)
        if num_speakers and num_speakers >= 1:
            n = min(num_speakers, len(embeddings))
            model = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
        else:
            model = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=config.DIARIZE_THRESHOLD,
            )
        return model.fit_predict(X).tolist()

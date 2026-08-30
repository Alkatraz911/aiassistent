"""Офлайн forced-alignment слов по аудио (wav2vec2, без токенов, без pyannote).

Тот же алгоритм, что внутри WhisperX (CTC forced alignment через torchaudio), но
минимальными зависимостями: torch + torchaudio + transformers. Русская модель
`wav2vec2-large-xlsr-53-russian` — публичная, без gated/токена.

Идея: для куска аудио и его текста выравниваем символы транскрипта по CTC-эмиссии
модели и собираем точные тайм-коды каждого слова.
"""
from __future__ import annotations

import numpy as np

from .. import config, device


class WordAligner:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._vocab: dict[str, int] = {}
        self._blank = 0
        self._delim = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._torch = torch
        self.device = device.resolve_torch_device(config.FINALIZE_DEVICE)
        name = config.ALIGN_MODEL
        self.processor = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).eval().to(self.device)
        self._vocab = {k.lower(): v for k, v in self.processor.tokenizer.get_vocab().items()}
        self._blank = self.processor.tokenizer.pad_token_id or 0
        # символ-разделитель слов в wav2vec2 (обычно "|")
        self._delim = self._vocab.get("|")

    def warmup(self) -> None:
        try:
            self._load()
        except Exception:
            pass

    def align(self, audio: np.ndarray, text: str, sample_rate: int) -> list[dict]:
        """Возвращает [{text,start,end,prob}] со временами в секундах внутри `audio`."""
        self._load()
        torch = self._torch
        import torchaudio.functional as AF

        # Для сопоставления с CTC-эмиссией текст нужен в нижнем регистре (словарь wav2vec2
        # строчный), но ОТДАВАТЬ наружу надо исходное написание: сегмент рисуется в протоколе
        # по словам, и подмена «Миша» на «миша» портила бы уже готовый текст. Оба списка —
        # результат одного и того же split(), поэтому индексы совпадают один в один.
        words = [w for w in text.lower().split() if w]
        originals = [w for w in text.split() if w]
        if not words or audio.size < sample_rate // 10:
            return []

        # targets: id символов; параллельно — индекс слова (или -1 для разделителя)
        targets: list[int] = []
        token_word: list[int] = []
        for wi, word in enumerate(words):
            if self._delim is not None and targets:
                targets.append(self._delim)
                token_word.append(-1)
            for ch in word:
                cid = self._vocab.get(ch)
                if cid is not None:
                    targets.append(cid)
                    token_word.append(wi)
        if not any(t >= 0 for t in token_word):
            return []

        with torch.inference_mode():
            inp = self.processor(audio, sampling_rate=sample_rate,
                                 return_tensors="pt").input_values.to(self.device)
            logits = self.model(inp).logits[0]            # (T, V)
            # Эмиссию считаем в float32 и возвращаем на CPU: сам forced_align ниже — это
            # динамическое программирование по (T x targets), на GPU оно не ускоряется, а вот
            # тяжёлый прямой проход wav2vec2 выше — ускоряется, ради него всё и затевалось.
            emission = torch.log_softmax(logits.float(), dim=-1).cpu()

        T = emission.size(0)
        if T < len([t for t in token_word if t >= 0]):
            return []  # аудио слишком короткое для такого текста

        tgt = torch.tensor([targets], dtype=torch.int32)
        try:
            aligned, scores = AF.forced_align(emission.unsqueeze(0), tgt, blank=self._blank)
            spans = AF.merge_tokens(aligned[0], scores[0].exp())
        except Exception:
            return []

        # merge_tokens даёт по одному спану на каждый target-токен
        sec_per_frame = (len(audio) / T) / sample_rate
        word_spans: dict[int, list] = {}
        for span, wi in zip(spans, token_word):
            if wi < 0:
                continue
            word_spans.setdefault(wi, []).append(span)

        out: list[dict] = []
        for wi, word in enumerate(words):
            sp = word_spans.get(wi)
            if not sp:
                continue
            start = min(s.start for s in sp) * sec_per_frame
            end = max(s.end for s in sp) * sec_per_frame
            prob = float(np.mean([float(s.score) for s in sp]))
            out.append({"text": originals[wi], "start": round(start, 3),
                        "end": round(end, 3), "prob": round(prob, 3)})
        return out

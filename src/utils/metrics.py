# ======================
# ./src/utils/metrics.py
# ======================

import math
from typing import List, Tuple

from sacrebleu.metrics import BLEU, CHRF
import sentencepiece as spm
import torch

from src.models.transformer import TranslationTransformer


# ==============================================================================
# SACREBLEU METRIC CONFIGURATION
# ==============================================================================
#
# These settings are intentionally explicit.
#
# They reproduce the effective behavior verified with SacreBLEU 2.6.0 for:
#
#     BLEU()
#
# and:
#
#     CHRF(word_order=2)
#
# The purpose of making them explicit is to prevent future changes in library
# defaults from silently modifying the experimental evaluation protocol.
# ==============================================================================


def _build_bleu_metric() -> BLEU:
    """
    Builds the corpus-level BLEU metric using the explicitly fixed
    evaluation configuration used in the experiments.
    """

    return BLEU(
        lowercase=False,
        force=False,
        tokenize="13a",
        smooth_method="exp",
        smooth_value=None,
        max_ngram_order=4,
        effective_order=False,
    )


def _build_chrf_metric() -> CHRF:
    """
    Builds the corpus-level chrF++ metric using the explicitly fixed
    evaluation configuration used in the experiments.

    word_order=2 corresponds to chrF++.
    """

    return CHRF(
        char_order=6,
        word_order=2,
        beta=2,
        lowercase=False,
        whitespace=False,
        eps_smoothing=False,
    )


# ==============================================================================
# GREEDY DECODING
# ==============================================================================


@torch.no_grad()
def greedy_decode(
        model: TranslationTransformer,
        src: torch.Tensor,
        sp: spm.SentencePieceProcessor,
        max_len: int = 150,
        device: torch.device = torch.device("cpu"),
) -> List[int]:
    """
    Greedy autoregressive decoding.

    At each decoding step, the token with the highest model probability
    is selected.

    Parameters
    ----------
    model:
        Trained TranslationTransformer.

    src:
        One tokenized source sequence.

    sp:
        SentencePiece processor used by the model.

    max_len:
        Maximum number of generated tokens after BOS.

    device:
        Device used for inference.

    Returns
    -------
    List[int]
        Generated token-ID sequence including BOS and,
        when generated, EOS.
    """

    model.eval()

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    if bos_id < 0:
        raise ValueError(
            "SentencePiece BOS token is not defined."
        )

    if eos_id < 0:
        raise ValueError(
            "SentencePiece EOS token is not defined."
        )

    if pad_id < 0:
        raise ValueError(
            "SentencePiece PAD token is not defined."
        )

    if max_len <= 0:
        raise ValueError(
            "max_len must be greater than zero."
        )

    # ==========================================================================
    # ENCODE SOURCE ONCE
    # ==========================================================================

    src = (
        src
        .unsqueeze(0)
        .to(device)
    )

    src_pad_mask = (
        src == pad_id
    )

    src_emb = model.embedding(
        src
    )

    src_emb = (
        src_emb
        * math.sqrt(
            model.d_model
        )
    )

    src_emb = model.pos_encoder(
        src_emb
    )

    # model.eval() disables dropout.
    src_emb = model.dropout(
        src_emb
    )

    memory = model.transformer.encoder(
        src_emb,
        src_key_padding_mask=src_pad_mask,
    )

    # ==========================================================================
    # AUTOREGRESSIVE GREEDY DECODING
    # ==========================================================================

    generated = [
        bos_id
    ]

    for _ in range(
        max_len
    ):

        tgt_tensor = torch.tensor(
            [generated],
            dtype=torch.long,
            device=device,
        )

        (
            _,
            tgt_causal_mask,
        ) = model.make_tgt_mask(
            tgt_tensor
        )

        tgt_emb = model.embedding(
            tgt_tensor
        )

        tgt_emb = (
            tgt_emb
            * math.sqrt(
                model.d_model
            )
        )

        tgt_emb = model.pos_encoder(
            tgt_emb
        )

        tgt_emb = model.dropout(
            tgt_emb
        )

        decoder_out = (
            model.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_pad_mask,
            )
        )

        logits = model.output_layer(
            decoder_out[
                :,
                -1,
                :
            ]
        )

        next_token = (
            logits
            .argmax(
                dim=-1
            )
            .item()
        )

        generated.append(
            next_token
        )

        if next_token == eos_id:
            break

    return generated


# ==============================================================================
# TOKEN-ID CLEANING / DETOKENIZATION
# ==============================================================================


def _prepare_ids_for_decoding(
        token_ids: List[int],
        sp: spm.SentencePieceProcessor,
) -> List[int]:
    """
    Removes sequence-control IDs before converting token IDs back to text.

    Decoding stops at the first EOS token.
    """

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    cleaned_ids = []

    for token_id in token_ids:

        if token_id == eos_id:
            break

        if token_id == bos_id:
            continue

        if token_id == pad_id:
            continue

        cleaned_ids.append(
            token_id
        )

    return cleaned_ids


def decode_token_ids(
        token_ids: List[int],
        sp: spm.SentencePieceProcessor,
) -> str:
    """
    Converts a token-ID sequence into normalized text while explicitly
    excluding BOS, EOS, and PAD control tokens.
    """

    cleaned_ids = (
        _prepare_ids_for_decoding(
            token_ids=token_ids,
            sp=sp,
        )
    )

    return sp.decode(
        cleaned_ids
    )


# ==============================================================================
# CORPUS EVALUATION
# ==============================================================================


@torch.no_grad()
def evaluate_metrics(
        model: TranslationTransformer,
        dataset,
        sp: spm.SentencePieceProcessor,
        device: torch.device,
        max_len: int = 150,
        return_predictions: bool = False,
) -> tuple:
    """
    Evaluates an NMT model using greedy decoding.

    Corpus-level BLEU and chrF++ are computed with SacreBLEU using
    explicitly fixed metric configurations.

    Parameters
    ----------
    model:
        Trained TranslationTransformer.

    dataset:
        Evaluation dataset yielding (src, tgt) tensors.

    sp:
        SentencePiece processor.

    device:
        Device used for inference.

    max_len:
        Maximum number of generated tokens after BOS.

    return_predictions:
        When False, returns only aggregate metrics and signatures.
        When True, additionally returns the exact reference and
        hypothesis strings used to compute the corpus-level metrics.

    Returns
    -------
    tuple
        If return_predictions is False:
            (
                BLEU score,
                chrF++ score,
                BLEU SacreBLEU signature,
                chrF++ SacreBLEU signature,
            )

        If return_predictions is True:
            (
                BLEU score,
                chrF++ score,
                BLEU SacreBLEU signature,
                chrF++ SacreBLEU signature,
                references,
                hypotheses,
            )
    """

    model.eval()

    hypotheses = []
    references = []

    # ==========================================================================
    # GENERATE TRANSLATIONS
    # ==========================================================================

    for idx in range(
        len(dataset)
    ):

        src, tgt = (
            dataset[idx]
        )

        # ----------------------------------------------------------------------
        # Reconstruct the exact reference text used for scoring.
        # Sequence-control tokens are removed explicitly.
        # ----------------------------------------------------------------------

        ref_text = decode_token_ids(
            token_ids=tgt.tolist(),
            sp=sp,
        )

        # ----------------------------------------------------------------------
        # Greedy decoding.
        # ----------------------------------------------------------------------

        pred_ids = greedy_decode(
            model=model,
            src=src,
            sp=sp,
            max_len=max_len,
            device=device,
        )

        pred_text = decode_token_ids(
            token_ids=pred_ids,
            sp=sp,
        )

        references.append(
            ref_text
        )

        hypotheses.append(
            pred_text
        )

    # ==========================================================================
    # SACREBLEU METRICS
    # ==========================================================================

    bleu_metric = (
        _build_bleu_metric()
    )

    chrf_metric = (
        _build_chrf_metric()
    )

    bleu_obj = (
        bleu_metric.corpus_score(
            hypotheses,
            [references],
        )
    )

    chrf_obj = (
        chrf_metric.corpus_score(
            hypotheses,
            [references],
        )
    )

    bleu_signature = str(
        bleu_metric.get_signature()
    )

    chrf_signature = str(
        chrf_metric.get_signature()
    )

    # ==========================================================================
    # RETURN SCORES + REPRODUCIBILITY SIGNATURES
    # ==========================================================================

    if return_predictions:

        return (
            bleu_obj.score,
            chrf_obj.score,
            bleu_signature,
            chrf_signature,
            references,
            hypotheses,
        )

    return (
        bleu_obj.score,
        chrf_obj.score,
        bleu_signature,
        chrf_signature,
    )
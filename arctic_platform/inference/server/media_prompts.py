"""Media prompts: vLLM ``TokensPrompt`` keys with ``multi_modal_data``.

A caller that embeds the server imports this module to learn whether a worker
takes media prompts (``SUPPORTS_MEDIA``). Media arrives as raw encoded bytes, one
placeholder token per item; ``decode_media`` turns them into what vLLM's engine takes, with
vLLM's own loaders, keeping the original bytes (``MediaWithBytes``) so that
vLLM's media hashes match.

``decodable_modalities`` says which modalities ``decode_media`` can decode here:
images always (Pillow), video only when the library of the codec the engine's
video loader decodes with is importable. ``decode_media`` refuses video bytes otherwise.

Importing this module does not import vLLM or any video library.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from arctic_platform.inference.server.token_prompts import check_token_tensor, is_token_tensor

SUPPORTS_MEDIA = True

__all__ = [
    "MEDIA_FORMATS",
    "MEDIA_PROMPT_KEYS",
    "SUPPORTS_MEDIA",
    "MediaDecodeError",
    "decodable_modalities",
    "decode_media",
    "is_media_prompt",
    "validate_media_prompt",
]

MEDIA_PROMPT_KEYS = frozenset(
    {"prompt_token_ids", "multi_modal_data", "mm_processor_kwargs", "multi_modal_uuids"}
)

# The encoded formats ``decode_media`` hands to vLLM's loaders, by modality, as
# (offset, leading bytes) signatures: PNG and JPEG images, and MP4 video (an
# ISO-BMFF file whose first box is ``ftyp`` with one of these major brands).
# Bytes in any other format are refused before a loader sees them, so Pillow's
# other plugins and the video codec's other demuxers never parse request bytes.
_MP4_BRANDS = (b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1")
MEDIA_FORMATS: dict[str, tuple[tuple[int, bytes], ...]] = {
    "image": ((0, b"\x89PNG\r\n\x1a\n"), (0, b"\xff\xd8\xff")),
    "video": tuple((4, b"ftyp" + brand) for brand in _MP4_BRANDS),
}


class MediaDecodeError(ValueError):
    """A media item's bytes could not be decoded.

    The message names only the modality, the item's index and either the
    decoder's exception type or "unsupported format", never the bytes: the
    decoder's own message may echo the input.
    """


# vLLM names two things in the ``video`` entry of ``media_io_kwargs``: the video
# loader (``video_backend``, else ``VLLM_VIDEO_LOADER_BACKEND``, else ``"opencv"``;
# an entry of its loader registry, such as ``opencv`` or a model's own loader)
# and the codec that loader decodes with (``backend``, else ``"opencv"``). The
# ``pynvvideocodec`` loader always decodes with PyNvVideoCodec. This is the
# library each codec imports; a codec not named here (``deepstream``, or any
# other) counts as not decodable.
_VIDEO_CODEC_MODULES = {"opencv": "cv2", "torchcodec": "torchcodec", "pynvvideocodec": "PyNvVideoCodec"}


def _video_codec(media_io_kwargs: dict[str, dict[str, Any]] | None) -> Any:
    video = (media_io_kwargs or {}).get("video") or {}
    loader = video.get("video_backend")
    if loader is None:
        loader = os.environ.get("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    if loader == "pynvvideocodec":
        return "pynvvideocodec"
    # Any other loader, a model's or one vLLM does not know, decodes with the codec.
    codec = video.get("backend")
    return "opencv" if codec is None else codec


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def decodable_modalities(media_io_kwargs: dict[str, dict[str, Any]] | None = None) -> frozenset[str]:
    """Return the modalities whose raw bytes ``decode_media`` can decode here.

    Always ``"image"``; ``"video"`` when the library of the video codec is
    importable. The codec is ``"pynvvideocodec"`` when the video loader (the
    ``video`` entry's ``video_backend``, else ``VLLM_VIDEO_LOADER_BACKEND``,
    else ``"opencv"``) is ``pynvvideocodec``; otherwise the ``video`` entry's
    ``backend``, else ``"opencv"``. The library is looked up, not imported.
    """
    codec = _video_codec(media_io_kwargs)
    module = _VIDEO_CODEC_MODULES.get(codec) if isinstance(codec, str) else None
    if module is not None and _importable(module):
        return frozenset({"image", "video"})
    return frozenset({"image"})


def is_media_prompt(prompt) -> bool:
    # The key, not the dict, makes a media prompt: any other dict is refused as before.
    return isinstance(prompt, dict) and "multi_modal_data" in prompt


def validate_media_prompt(prompt: dict) -> dict:
    """Check a media prompt's shape and return a copy with ``None`` keys dropped.

    The token ids are the unexpanded ones: vLLM's processor replaces each
    placeholder with the item's tokens. A 1-D token tensor becomes a list. The
    media itself is vLLM's to check.
    """
    unknown = prompt.keys() - MEDIA_PROMPT_KEYS
    if unknown:
        raise ValueError(f"Unknown media prompt keys: {sorted(unknown)}")
    token_ids = prompt.get("prompt_token_ids")
    if is_token_tensor(token_ids):
        token_ids = check_token_tensor(token_ids).tolist()
    if not isinstance(token_ids, list) or not 0 < len(token_ids) <= 131072 or any(
        type(token) is not int or not 0 <= token < 2**31 for token in token_ids
    ):
        raise ValueError("Media prompt_token_ids must be 1..131072 nonnegative token IDs")
    media = prompt.get("multi_modal_data")
    if not isinstance(media, dict) or not media:
        raise ValueError("A media prompt needs a non-empty multi_modal_data dict; send token ids as a list")
    for key in ("mm_processor_kwargs", "multi_modal_uuids"):
        if prompt.get(key) is not None and not isinstance(prompt[key], dict):
            raise ValueError(f"Media prompt {key} must be a dict")
    checked = {key: value for key, value in prompt.items() if value is not None}
    checked["prompt_token_ids"] = list(token_ids)
    return checked


def _is_bytes(item: Any) -> bool:
    return isinstance(item, (bytes, bytearray, memoryview))


def _is_allowed_format(modality: str, item: Any) -> bool:
    head = bytes(item[:16])
    return any(head[offset : offset + len(magic)] == magic for offset, magic in MEDIA_FORMATS[modality])


def decode_media(prompt: dict, media_io_kwargs: dict[str, dict[str, Any]] | None = None) -> dict:
    """Return *prompt* with raw encoded image and video bytes decoded for vLLM.

    Each bytes item becomes a ``MediaWithBytes`` made by vLLM's ``ImageMediaIO`` or
    ``VideoMediaIO`` with the engine's ``media_io_kwargs``; items that are not
    bytes (already decoded) pass through. Before anything is decoded, image and
    video bytes not in a ``MEDIA_FORMATS`` format raise ``MediaDecodeError``
    (``unsupported format``). An item whose ``load_bytes`` fails raises
    ``MediaDecodeError``, unchained, and no later item is decoded. Video bytes when
    ``"video"`` is not in ``decodable_modalities`` raise ``MediaDecodeError``
    (``video decoding is not available``) before any loader is built. Video when
    vLLM's video loader cannot be imported, and bytes of any other modality, raise
    ``ValueError``.
    """
    media = prompt["multi_modal_data"]
    if not any(_is_bytes(item) for items in media.values() for item in (items if isinstance(items, list) else [items])):
        return prompt
    videos = media.get("video")
    if any(_is_bytes(item) for item in (videos if isinstance(videos, list) else [videos])) and (
        "video" not in decodable_modalities(media_io_kwargs)
    ):
        raise MediaDecodeError("video decoding is not available")
    for modality, items in media.items():
        if modality not in MEDIA_FORMATS:
            continue
        for index, item in enumerate(items if isinstance(items, list) else [items]):
            if _is_bytes(item) and not _is_allowed_format(modality, item):
                raise MediaDecodeError(f"could not decode {modality} {index} (unsupported format)")
    media_io_kwargs = media_io_kwargs or {}
    loaders: dict[str, Any] = {}

    def loader(modality: str):
        if modality not in loaders:
            loaders[modality] = _make_loader(modality, media_io_kwargs, loaders.get("image"))
        return loaders[modality]

    decoded: dict[str, Any] = {}
    for modality, items in media.items():
        single = not isinstance(items, list)
        out = []
        for index, item in enumerate([items] if single else items):
            if not _is_bytes(item):
                out.append(item)
                continue
            media_io = loader(modality)
            failed = None
            try:
                out.append(media_io.load_bytes(bytes(item)))
            except Exception as error:
                failed = type(error).__name__
            if failed is not None:
                # Raised outside the handler, so the decoder's exception is not even the context.
                raise MediaDecodeError(f"could not decode {modality} {index} ({failed})") from None
        decoded[modality] = out[0] if single else out
    return {**prompt, "multi_modal_data": decoded}


def _make_loader(modality: str, media_io_kwargs: dict[str, dict[str, Any]], image_io: Any) -> Any:
    if modality == "image":
        from vllm.multimodal.media.image import ImageMediaIO

        return ImageMediaIO(**ImageMediaIO.merge_kwargs(media_io_kwargs.get("image"), None))
    if modality == "video":
        try:
            from vllm.multimodal.media.image import ImageMediaIO
            from vllm.multimodal.media.video import VideoMediaIO

            image_io = image_io or ImageMediaIO(**ImageMediaIO.merge_kwargs(media_io_kwargs.get("image"), None))
            return VideoMediaIO(image_io, **VideoMediaIO.merge_kwargs(media_io_kwargs.get("video"), None))
        except ImportError as error:
            raise ValueError(f"Cannot decode video: this worker has no vLLM video decoder ({error})") from error
    raise ValueError(f"Cannot decode raw {modality} bytes; send {modality} items already decoded")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI provider abstraction for the GIMP AI Plugin.

Pure Python with no GIMP dependencies (same rule as coordinate_utils.py), so
providers can be unit tested without GIMP installed.

A provider owns everything that is specific to one AI service: endpoint URLs,
model names, request construction, response parsing, size selection and error
wording. The plugin keeps the transport (GimpAIPlugin._make_url_request, with
its SSL fallback and threading) and all GIMP-side work.

Adding a provider means adding a Provider subclass and registering it in
PROVIDERS - no changes to the image processing pipeline.
"""

import base64
import json
import urllib.request
import uuid

# Provider used when the config file does not name one. Existing config files
# have no provider key at all, so they keep working unchanged.
DEFAULT_PROVIDER = "openai"


class ProviderError(Exception):
    """Raised when a response cannot be parsed into image data."""


def create_multipart_data(fields, files):
    """Create multipart form data for file upload - supports image arrays

    Args:
        fields: Dict of text form fields
        files: Dict of file fields, each a (filename, data, content_type)
            tuple, or a list of such tuples for the "image" array field

    Returns:
        tuple: (body_bytes, boundary_string)
    """
    boundary = uuid.uuid4().hex
    body = b""

    # Add text fields
    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()

    # Add file fields - handle both single files and arrays
    for key, file_data in files.items():
        if key == "image" and isinstance(file_data, list):
            # Handle image array for composite mode - use image[] array syntax
            for i, (filename, data, content_type) in enumerate(file_data):
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="image[]"; filename="{filename}"\r\n'.encode()
                body += f"Content-Type: {content_type}\r\n\r\n".encode()
                body += data
                body += b"\r\n"
            print(f"DEBUG: Added {len(file_data)} images to multipart data")
        else:
            # Handle single file (like mask or single image)
            filename, data, content_type = file_data
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
            body += f"Content-Type: {content_type}\r\n\r\n".encode()
            body += data
            body += b"\r\n"

    # End boundary
    body += f"--{boundary}--\r\n".encode()

    return body, boundary


def parse_b64_json_response(body):
    """Decode an OpenAI-style {"data": [{"b64_json": ...}]} response.

    Venice's /images/generations endpoint is OpenAI compatible, so both
    providers share this.

    Args:
        body: Response body as bytes

    Returns:
        bytes: Decoded image data

    Raises:
        ProviderError: If the response contains no usable image
    """
    response_data = json.loads(body.decode("utf-8"))

    if "data" not in response_data or len(response_data["data"]) == 0:
        raise ProviderError("No data in API response")

    result_data = response_data["data"][0]
    if "b64_json" not in result_data:
        raise ProviderError("No image data in response")

    return base64.b64decode(result_data["b64_json"])


# Names for the nine cells of a 3x3 grid, indexed [row][column]
_REGION_NAMES = (
    ("the top left", "the top", "the top right"),
    ("the left", "the centre", "the right"),
    ("the bottom left", "the bottom", "the bottom right"),
)

# A selection covering this much of the image is treated as a full-frame edit
_FULL_FRAME_COVERAGE = 0.8


def describe_selection_position(sel_bounds, img_width, img_height):
    """Describe where a selection sits, as a phrase like "the centre".

    Providers with no mask parameter cannot be told which region to edit, so
    the region has to be named in the prompt instead. Without it the model
    composes for the whole frame and the layer mask clips whatever falls
    outside the selection - a figure asked for in a small selection comes back
    cut off at its edge.

    Args:
        sel_bounds: (x1, y1, x2, y2) selection bounds
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        str or None: A region name, or None when the selection covers most of
            the image and naming a region would be misleading
    """
    if not sel_bounds or len(sel_bounds) != 4:
        return None
    if not img_width or not img_height:
        return None

    x1, y1, x2, y2 = sel_bounds
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return None

    # A near-full-frame selection has no meaningful position
    coverage = (width * height) / float(img_width * img_height)
    if coverage >= _FULL_FRAME_COVERAGE:
        return None

    centre_x = (min(x1, x2) + width / 2.0) / float(img_width)
    centre_y = (min(y1, y2) + height / 2.0) / float(img_height)

    column = min(int(centre_x * 3), 2)
    row = min(int(centre_y * 3), 2)

    return _REGION_NAMES[row][column]


class Provider:
    """Base class for AI image providers.

    Subclasses describe themselves through the class attributes below and
    implement the four request/response methods.
    """

    # Identity - name is the config section key, label is shown to users
    name = ""
    label = ""

    # Credentials
    api_key_env = ""
    key_placeholder = ""

    # Largest number of images the edit endpoint accepts in one call.
    # None means the limit is unknown and should not be enforced locally.
    MAX_INPUT_IMAGES = 1

    # False when the API has no mask parameter, so the selection can only be
    # applied on the GIMP side after the result comes back
    requires_mask = True

    # Shown in the settings dialog for providers that are not production ready
    experimental = False

    # Name given to layers created from generated images
    generated_layer_name = "AI Generated"

    def __init__(self, config=None):
        """
        Args:
            config: Plugin config dict (may be None)
        """
        self.config = config or {}

    @property
    def settings(self):
        """This provider's section of the config, or an empty dict."""
        return self.config.get(self.name) or {}

    @property
    def max_input_images(self):
        """Largest number of images the edit endpoint accepts, or None."""
        return self.MAX_INPUT_IMAGES

    def get_api_key(self, environ):
        """Get the API key for this provider from config or environment.

        Args:
            environ: Environment mapping (normally os.environ)

        Returns:
            str or None: The API key, or None if not configured
        """
        api_key = self.settings.get("api_key")
        if api_key:
            return api_key

        api_key = environ.get(self.api_key_env)
        if api_key:
            return api_key

        return None

    def missing_key_message(self):
        """User-facing message shown when no API key is configured."""
        return (
            f"No {self.label} API key found!\n\nPlease set your API key in:\n"
            f"- config.json file\n- {self.api_key_env} environment variable"
        )

    # Subclass responsibilities

    def resolve_generation_size(self, size):
        """Resolve a requested generation size to one the API accepts."""
        raise NotImplementedError

    def build_generation_request(self, prompt, size, api_key):
        """Build the image generation request.

        Returns:
            urllib.request.Request: Ready to pass to the plugin's transport
        """
        raise NotImplementedError

    def parse_generation_response(self, body):
        """Parse a generation response body into raw image bytes.

        Args:
            body: Response body as bytes

        Returns:
            bytes: Decoded image data

        Raises:
            ProviderError: If the response contains no usable image
        """
        raise NotImplementedError

    def build_edit_request(self, image_bytes, mask_bytes, prompt, size, api_key):
        """Build the image edit request.

        Args:
            image_bytes: PNG bytes, or a list of PNG bytes for composite mode
            mask_bytes: PNG mask bytes, or None
            prompt: Edit prompt
            size: Size string such as "1536x1024"
            api_key: API key

        Returns:
            urllib.request.Request: Ready to pass to the plugin's transport
        """
        raise NotImplementedError

    def parse_edit_response(self, body):
        """Parse an edit response body into raw image bytes.

        Args:
            body: Response body as bytes

        Returns:
            bytes: Decoded image data

        Raises:
            ProviderError: If the response contains no usable image
        """
        raise NotImplementedError

    def format_http_error(self, code, body):
        """Format an HTTP error for display to the user."""
        return f"{self.label} API error {code}: {body[:200]}"

    def add_region_hint(self, prompt, region):
        """Name the edit region in the prompt, for providers with no mask.

        Providers that accept a mask ignore this and return the prompt as the
        user typed it.
        """
        return prompt

    # Optional: model discovery

    def build_models_request(self, kind):
        """Build a request listing available models, or None if unsupported.

        Args:
            kind: "generation" or "edit"

        Returns:
            urllib.request.Request or None
        """
        return None

    def parse_models_response(self, body):
        """Parse a model listing response.

        Returns:
            list: Dicts with at least an "id" key
        """
        return []

    def generation_models(self):
        """Model ids offered for generation before any live refresh."""
        return []

    def edit_models(self):
        """Model ids offered for editing before any live refresh."""
        return []


class OpenAIProvider(Provider):
    """OpenAI gpt-image-1 - generations and edits endpoints."""

    name = "openai"
    label = "OpenAI"

    api_key_env = "OPENAI_API_KEY"
    key_placeholder = "sk-proj-..."

    MAX_INPUT_IMAGES = 16

    generated_layer_name = "GPT-Image Generated"

    MODEL = "gpt-image-1"
    GENERATION_URL = "https://api.openai.com/v1/images/generations"
    EDIT_URL = "https://api.openai.com/v1/images/edits"

    # Used when the caller asks for "auto" rather than a specific shape
    DEFAULT_GENERATION_SIZE = "1536x1024"
    DEFAULT_EDIT_SIZE = "1024x1024"

    def resolve_generation_size(self, size):
        """OpenAI accepts the plugin's shape strings directly."""
        if not size or size == "auto":
            return self.DEFAULT_GENERATION_SIZE
        return size

    def build_generation_request(self, prompt, size, api_key):
        data = {
            "model": self.MODEL,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": "high",
        }

        json_data = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(self.GENERATION_URL, data=json_data)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        return req

    def parse_generation_response(self, body):
        return parse_b64_json_response(body)

    def build_edit_request(self, image_bytes, mask_bytes, prompt, size, api_key):
        fields = {
            "model": self.MODEL,
            "prompt": prompt,
            "n": "1",
            "quality": "high",
            "size": size if size else self.DEFAULT_EDIT_SIZE,
            "moderation": "low",  # Less restrictive filtering
            "input_fidelity": "high",  # High fidelity for better results
        }

        files = {}
        if isinstance(image_bytes, list):
            files["image"] = [
                (f"image_{i}.png", data, "image/png")
                for i, data in enumerate(image_bytes)
            ]
        else:
            files["image"] = ("image.png", image_bytes, "image/png")

        if mask_bytes:
            files["mask"] = ("mask.png", mask_bytes, "image/png")

        body, boundary = create_multipart_data(fields, files)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "GIMP-AI-Plugin/1.0",
        }

        return urllib.request.Request(
            self.EDIT_URL, data=body, headers=headers, method="POST"
        )

    def parse_edit_response(self, body):
        return parse_b64_json_response(body)

    def format_http_error(self, code, body):
        return f"GPT-Image-1 API error {code}: {body[:200]}"


class VeniceProvider(Provider):
    """Venice.ai - OpenAI compatible generation plus experimental edit endpoints.

    Differences from OpenAI that shape this implementation:

    - The edit endpoints return raw image bytes, not JSON.
    - There is no mask parameter anywhere. /image/generate's inpaint field was
      disabled in May 2025 and /image/{edit,multi-edit} never had one, so the
      GIMP selection can only be applied locally, after the result arrives.
    - Edits are sized by aspect ratio and a resolution tier, not by pixels, so
      the result comes back at the model's own dimensions.
    - The edit endpoints use different model field names: /image/edit takes
      "model", /image/multi-edit takes "modelId".

    Verified against https://api.venice.ai/api/v1/swagger.yaml, 2026-09-18.
    """

    name = "venice"
    label = "Venice"

    api_key_env = "VENICE_API_KEY"
    key_placeholder = "Venice API key"

    experimental = True
    requires_mask = False  # no mask parameter on any Venice endpoint
    generated_layer_name = "Venice Generated"

    BASE_URL = "https://api.venice.ai/api/v1"
    GENERATION_URL = f"{BASE_URL}/images/generations"
    EDIT_URL = f"{BASE_URL}/image/edit"
    MULTI_EDIT_URL = f"{BASE_URL}/image/multi-edit"
    MODELS_URL = f"{BASE_URL}/models"

    DEFAULT_MODEL = "venice-sd35"
    DEFAULT_EDIT_MODEL = "firered-image-edit"

    # Resolution tiers and aspect ratios are per-model and most edit models
    # support neither. Sending an unsupported value is a 400, so both are
    # omitted unless the model is known to accept them - the input image
    # already carries the shape, which Venice infers from when told nothing.
    RESOLUTION_MODEL_DEFAULT = ""

    # Most edit models declare maxInputImages 6; the settings dialog refreshes
    # this from the live catalogue, and the API rejects anything above its own
    # limit with a 400 that we surface verbatim.
    DEFAULT_MAX_INPUT_IMAGES = 6

    # The generation endpoint takes the same size strings as OpenAI. Venice's
    # native /image/generate caps width and height at 1280, which is why this
    # uses the OpenAI compatible endpoint instead.
    DEFAULT_GENERATION_SIZE = "1536x1024"

    # The three shapes the plugin produces map onto these aspect ratios. Only
    # sent when the selected model's catalogue entry lists the value.
    ASPECT_RATIOS = {
        "1024x1024": "1:1",
        "1536x1024": "3:2",
        "1024x1536": "2:3",
    }

    # Offered in the settings dialog before a live refresh. Both dropdowns
    # accept free text, so these are a starting point, not a whitelist.
    FALLBACK_MODELS = [
        "venice-sd35",
        "z-image-turbo",
        "qwen-image-3",
        "qwen-image-3-pro",
        "flux-2-pro",
        "flux-2-max",
        "nano-banana-2",
        "nano-banana-pro",
        "seedream-v5-pro",
        "hunyuan-image-v3",
        "recraft-v4",
    ]

    FALLBACK_EDIT_MODELS = [
        "firered-image-edit",
        "qwen-image-3-edit",
        "qwen-image-2-edit",
        "gpt-image-2-edit",
        "gpt-image-2-5-flare-edit",
        "nano-banana-2-edit",
        "nano-banana-pro-edit",
        "seedream-v5-pro-edit",
        "flux-2-max-edit",
        "muse-image-edit",
    ]

    # Venice model types: generation models and edit ("inpaint") models are
    # returned by separate catalogue queries.
    MODEL_TYPES = {"generation": "image", "edit": "inpaint"}

    @property
    def max_input_images(self):
        """Limit for the selected edit model, refreshed from the catalogue."""
        limit = self.settings.get("max_input_images")
        if limit is None:
            return self.DEFAULT_MAX_INPUT_IMAGES
        try:
            return int(limit)
        except (TypeError, ValueError):
            return self.DEFAULT_MAX_INPUT_IMAGES

    def _model(self):
        return self.settings.get("model") or self.DEFAULT_MODEL

    def _edit_model(self):
        return self.settings.get("edit_model") or self.DEFAULT_EDIT_MODEL

    def _safe_mode(self):
        """Whether Venice should blur content it classifies as adult.

        Venice defaults this on. The OpenAI path has always sent moderation
        "low", so without this setting the same plugin would filter more
        heavily on one provider than the other for no stated reason.
        """
        return bool(self.settings.get("safe_mode", True))

    def _resolution(self):
        """Resolution tier to send, or None to let the model decide.

        Most edit models have no resolution tiers, and sending one to a model
        that lacks them is rejected outright, so nothing is sent unless the
        user picked a tier explicitly.
        """
        resolution = self.settings.get("resolution")
        if not resolution:
            return None

        # Only send a tier the catalogue confirms this model takes. Unknown
        # counts as unsupported: most edit models have no tiers, and a stale
        # value left in the config would otherwise fail every request.
        supported = self.settings.get("edit_resolutions")
        if not supported or resolution not in supported:
            print(
                f"DEBUG: Dropping resolution {resolution} - not confirmed for "
                f"{self._edit_model()}. Refresh the model list in Settings."
            )
            return None

        return resolution

    def _aspect_ratio(self, size):
        """Aspect ratio to send, or None to let Venice infer it.

        Venice infers the ratio from the input image when told nothing, and
        the image the plugin sends is already padded to the target shape - so
        omitting this is both safe and correct. It is only sent when the
        catalogue confirms the model accepts the value, because the supported
        list differs per model and several do not accept "auto" at all.
        """
        ratio = self.ASPECT_RATIOS.get(size)
        if not ratio:
            return None

        supported = self.settings.get("edit_aspect_ratios")
        if not supported or ratio not in supported:
            return None

        return ratio

    def _json_request(self, url, data, api_key):
        json_data = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=json_data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", "GIMP-AI-Plugin/1.0")
        return req

    def resolve_generation_size(self, size):
        if not size or size == "auto":
            return self.DEFAULT_GENERATION_SIZE
        return size

    def build_generation_request(self, prompt, size, api_key):
        data = {
            "model": self._model(),
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
            "output_format": "png",
            # The compatible endpoint spells safe mode "moderation"
            "moderation": "auto" if self._safe_mode() else "low",
        }
        return self._json_request(self.GENERATION_URL, data, api_key)

    def parse_generation_response(self, body):
        return parse_b64_json_response(body)

    def build_edit_request(self, image_bytes, mask_bytes, prompt, size, api_key):
        """Build an edit request.

        mask_bytes is accepted for interface compatibility and deliberately
        ignored: no Venice endpoint takes a mask. The caller applies the
        selection as a GIMP layer mask once the result is composited.
        """
        data = {
            "prompt": prompt,
            "output_format": "png",
            "safe_mode": self._safe_mode(),
        }

        # Both are model-specific and omitted unless known to be accepted
        aspect_ratio = self._aspect_ratio(size)
        if aspect_ratio:
            data["aspect_ratio"] = aspect_ratio

        resolution = self._resolution()
        if resolution:
            data["resolution"] = resolution

        if isinstance(image_bytes, list):
            # Layer composite - first image is the base, the rest are layers
            data["modelId"] = self._edit_model()
            data["images"] = [
                base64.b64encode(data_bytes).decode("ascii")
                for data_bytes in image_bytes
            ]
            url = self.MULTI_EDIT_URL
        else:
            data["model"] = self._edit_model()
            data["image"] = base64.b64encode(image_bytes).decode("ascii")
            url = self.EDIT_URL

        return self._json_request(url, data, api_key)

    def add_region_hint(self, prompt, region):
        """Append the edit region, since Venice cannot be sent a mask.

        "add a dwarf sitting" with a centre selection becomes "add a dwarf
        sitting, in the centre of the image". Without it the model composes for
        the whole frame and the layer mask clips whatever falls outside the
        selection.
        """
        if not region or not prompt or not prompt.strip():
            return prompt

        if region in prompt.lower():
            return prompt

        return f"{prompt.rstrip().rstrip(',.')}, in {region} of the image"

    def parse_edit_response(self, body):
        """Venice edit endpoints answer with raw image bytes on success."""
        if not body:
            return self._raise_body_error(body)

        # Errors still arrive as JSON, even from an endpoint that normally
        # returns binary, so anything JSON-shaped is a failure.
        if body[:1] in (b"{", b"["):
            self._raise_body_error(body)

        return body

    def _raise_body_error(self, body):
        raise ProviderError(self._error_text(body) or "Empty response from Venice")

    def _error_text(self, body):
        """Pull a human readable message out of a Venice error body.

        Venice returns {"error": "<string>"}, not OpenAI's nested
        {"error": {"message": ...}}, so both shapes are handled.
        """
        if not body:
            return ""

        if isinstance(body, bytes):
            try:
                body = body.decode("utf-8", "replace")
            except Exception:
                return ""

        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            return body.strip()[:200]

        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:200]
        if error:
            return str(error)[:200]

        details = payload.get("details") if isinstance(payload, dict) else None
        if details:
            return str(details)[:200]

        return str(payload)[:200]

    # Venice status codes that deserve an explanation rather than a raw dump
    HTTP_HINTS = {
        401: "Venice rejected the API key. Check it in Settings.",
        402: "Venice account has insufficient balance.",
        415: "Venice rejected the request format.",
        429: "Venice rate limit reached. Wait a moment and try again.",
        500: "Venice failed to process the request. Try again.",
        503: "The Venice model is at capacity. Try again or choose another model.",
    }

    def format_http_error(self, code, body):
        detail = self._error_text(body)
        hint = self.HTTP_HINTS.get(code)

        if hint and detail:
            return f"Venice API error {code}: {hint} ({detail})"
        if hint:
            return f"Venice API error {code}: {hint}"
        return f"Venice API error {code}: {detail}"

    # Model discovery - the catalogue is public, so no API key is sent

    def build_models_request(self, kind):
        model_type = self.MODEL_TYPES.get(kind)
        if not model_type:
            return None

        req = urllib.request.Request(f"{self.MODELS_URL}?type={model_type}")
        req.add_header("User-Agent", "GIMP-AI-Plugin/1.0")
        return req

    def parse_models_response(self, body):
        """Parse GET /models into id plus the constraints the plugin uses.

        The endpoint documentation refers to capabilities.maxInputImages, but
        the live API returns an empty capabilities object for image models and
        puts the limit in model_spec.constraints, so both are checked.
        """
        payload = json.loads(body.decode("utf-8"))
        models = []

        for entry in payload.get("data") or []:
            model_id = entry.get("id")
            if not model_id:
                continue

            spec = entry.get("model_spec") or {}
            constraints = spec.get("constraints") or {}
            capabilities = spec.get("capabilities") or {}

            max_inputs = constraints.get("maxInputImages")
            if max_inputs is None:
                max_inputs = capabilities.get("maxInputImages")
            if max_inputs is None and constraints.get("combineImages") is False:
                max_inputs = 1

            models.append(
                {
                    "id": model_id,
                    "max_input_images": max_inputs,
                    "aspect_ratios": constraints.get("aspectRatios") or [],
                    "resolutions": constraints.get("resolutions") or [],
                    "prompt_limit": constraints.get("promptCharacterLimit"),
                }
            )

        return models

    def generation_models(self):
        return list(self.FALLBACK_MODELS)

    def edit_models(self):
        return list(self.FALLBACK_EDIT_MODELS)


PROVIDERS = {
    OpenAIProvider.name: OpenAIProvider,
    VeniceProvider.name: VeniceProvider,
}


def get_provider_name(config):
    """Get the configured provider name, falling back to the default.

    Accepts both "provider" and the legacy "api_provider" key used by
    config.json.example. Unknown names fall back to the default rather than
    failing, so a typo cannot lock the user out of the plugin.

    Args:
        config: Plugin config dict (may be None)

    Returns:
        str: A key of PROVIDERS
    """
    if not config:
        return DEFAULT_PROVIDER

    name = config.get("provider") or config.get("api_provider") or DEFAULT_PROVIDER
    name = str(name).strip().lower()

    if name not in PROVIDERS:
        print(f"DEBUG: Unknown provider '{name}', using {DEFAULT_PROVIDER}")
        return DEFAULT_PROVIDER

    return name


def get_provider(config):
    """Get a provider instance for the given config.

    Args:
        config: Plugin config dict (may be None)

    Returns:
        Provider: An instance of the configured provider
    """
    return PROVIDERS[get_provider_name(config)](config)

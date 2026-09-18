#!/usr/bin/env python3
"""
Test suite for the AI provider abstraction.

Covers provider selection, credential lookup, request construction and
response parsing. No network access: HTTP responses are canned, so these
tests run anywhere without an API key.
"""

import base64
import json
import sys
import os

# Add parent directory to path so we can import ai_providers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_providers import (
    DEFAULT_PROVIDER,
    describe_selection_position,
    OpenAIProvider,
    ProviderError,
    VeniceProvider,
    create_multipart_data,
    get_provider,
    get_provider_name,
)

# A tiny valid-looking PNG header, enough to prove bytes survive the round trip
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake image data"
MASK_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake mask data"


def test_provider_selection():
    """Test that the configured provider is resolved correctly."""
    print("=== Testing Provider Selection ===")

    # No config at all - existing installs have no provider key
    assert get_provider_name(None) == DEFAULT_PROVIDER
    assert get_provider_name({}) == DEFAULT_PROVIDER
    print("✓ Missing config falls back to the default provider")

    # Explicit selection
    assert get_provider_name({"provider": "openai"}) == "openai"
    assert get_provider_name({"provider": "OpenAI"}) == "openai"
    assert get_provider_name({"provider": " openai "}) == "openai"
    print("✓ Explicit provider names are normalised")

    # Legacy key from config.json.example
    assert get_provider_name({"api_provider": "openai"}) == "openai"
    print("✓ Legacy api_provider key is accepted")

    assert get_provider_name({"provider": "venice"}) == "venice"
    assert isinstance(get_provider({"provider": "venice"}), VeniceProvider)
    print("✓ Venice can be selected")

    # Unknown names must not lock the user out
    assert get_provider_name({"provider": "not-a-provider"}) == DEFAULT_PROVIDER
    print("✓ Unknown provider names fall back to the default")

    provider = get_provider({})
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"
    assert provider.label == "OpenAI"
    print("✓ get_provider returns a provider instance")


def test_api_key_lookup():
    """Test API key resolution from config and environment."""
    print("\n=== Testing API Key Lookup ===")

    environ = {"OPENAI_API_KEY": "env-key", "VENICE_API_KEY": "venice-env-key"}

    # Config takes priority over the environment
    provider = OpenAIProvider({"openai": {"api_key": "config-key"}})
    assert provider.get_api_key(environ) == "config-key"
    print("✓ Config key takes priority")

    # Environment is the fallback
    assert OpenAIProvider({}).get_api_key(environ) == "env-key"
    assert OpenAIProvider({"openai": {}}).get_api_key(environ) == "env-key"
    assert OpenAIProvider({"openai": {"api_key": None}}).get_api_key(environ) == "env-key"
    print("✓ Environment variable is used as fallback")

    # Nothing configured
    assert OpenAIProvider({}).get_api_key({}) is None
    assert OpenAIProvider(None).get_api_key({}) is None
    print("✓ Missing key returns None")

    # Each provider reads its own section and its own env var
    both = {"openai": {"api_key": "oa"}, "venice": {"api_key": "vn"}}
    assert OpenAIProvider(both).get_api_key({}) == "oa"
    assert VeniceProvider(both).get_api_key({}) == "vn"
    assert VeniceProvider({}).get_api_key(environ) == "venice-env-key"
    print("✓ Providers are isolated from each other's credentials")

    # The message names the provider and its env var
    message = OpenAIProvider({}).missing_key_message()
    assert "OpenAI" in message
    assert "OPENAI_API_KEY" in message

    message = VeniceProvider({}).missing_key_message()
    assert "Venice" in message
    assert "VENICE_API_KEY" in message
    print("✓ Missing key message names the provider and env var")


def test_generation_request():
    """Test that generation requests match the OpenAI images API."""
    print("\n=== Testing Generation Request ===")

    provider = OpenAIProvider()

    # "auto" resolves to the default landscape shape
    assert provider.resolve_generation_size("auto") == "1536x1024"
    assert provider.resolve_generation_size(None) == "1536x1024"
    assert provider.resolve_generation_size("1024x1536") == "1024x1536"
    print("✓ Generation size resolution is correct")

    req = provider.build_generation_request("a red dragon", "1024x1024", "sk-test")

    assert req.full_url == "https://api.openai.com/v1/images/generations"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("Authorization") == "Bearer sk-test"
    print("✓ Generation URL and headers are correct")

    body = json.loads(req.data.decode("utf-8"))
    assert body == {
        "model": "gpt-image-1",
        "prompt": "a red dragon",
        "n": 1,
        "size": "1024x1024",
        "quality": "high",
    }
    print("✓ Generation request body is correct")


def test_generation_response_parsing():
    """Test parsing of generation responses, including failure modes."""
    print("\n=== Testing Generation Response Parsing ===")

    provider = OpenAIProvider()

    body = json.dumps(
        {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
    ).encode("utf-8")
    assert provider.parse_generation_response(body) == PNG_BYTES
    print("✓ Base64 image data is decoded")

    # Empty and missing data arrays
    for payload in ({"data": []}, {}):
        try:
            provider.parse_generation_response(json.dumps(payload).encode("utf-8"))
            raise AssertionError(f"Expected ProviderError for {payload}")
        except ProviderError as e:
            assert str(e) == "No data in API response"
    print("✓ Missing data raises ProviderError")

    # Present but unusable entry
    try:
        provider.parse_generation_response(
            json.dumps({"data": [{"revised_prompt": "x"}]}).encode("utf-8")
        )
        raise AssertionError("Expected ProviderError for missing b64_json")
    except ProviderError as e:
        assert str(e) == "No image data in response"
    print("✓ Missing image data raises ProviderError")


def test_edit_request_single_image():
    """Test inpainting requests - one image plus a mask."""
    print("\n=== Testing Edit Request (Single Image) ===")

    provider = OpenAIProvider()
    req = provider.build_edit_request(
        PNG_BYTES, MASK_BYTES, "blue sky", "1536x1024", "sk-test"
    )

    assert req.full_url == "https://api.openai.com/v1/images/edits"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer sk-test"
    assert req.get_header("User-agent") == "GIMP-AI-Plugin/1.0"

    content_type = req.get_header("Content-type")
    assert content_type.startswith("multipart/form-data; boundary=")
    print("✓ Edit URL, method and headers are correct")

    body = req.data
    assert b'name="model"' in body and b"gpt-image-1" in body
    assert b'name="prompt"' in body and b"blue sky" in body
    assert b'name="size"' in body and b"1536x1024" in body
    assert b'name="quality"' in body and b"high" in body
    assert b'name="moderation"' in body and b"low" in body
    assert b'name="input_fidelity"' in body
    print("✓ Edit form fields are present")

    assert b'name="image"; filename="image.png"' in body
    assert b'name="mask"; filename="mask.png"' in body
    assert PNG_BYTES in body
    assert MASK_BYTES in body
    print("✓ Image and mask parts carry the original bytes")

    # Size falls back to square when not supplied
    fallback = provider.build_edit_request(PNG_BYTES, MASK_BYTES, "x", None, "k")
    assert b"1024x1024" in fallback.data
    print("✓ Missing size falls back to 1024x1024")


def test_edit_request_image_array():
    """Test layer composite requests - several images, optional mask."""
    print("\n=== Testing Edit Request (Image Array) ===")

    provider = OpenAIProvider()
    layers = [b"\x89PNG-base", b"\x89PNG-upper-1", b"\x89PNG-upper-2"]

    req = provider.build_edit_request(layers, None, "blend these", "1024x1024", "k")
    body = req.data

    # Array mode uses the image[] field name for every input
    assert body.count(b'name="image[]"') == 3
    for i, layer in enumerate(layers):
        assert f'filename="image_{i}.png"'.encode() in body
        assert layer in body
    assert b'name="mask"' not in body
    print("✓ Array mode sends every layer as image[] with no mask")

    # A mask is included when supplied
    masked = provider.build_edit_request(layers, MASK_BYTES, "blend", "1024x1024", "k")
    assert b'name="mask"; filename="mask.png"' in masked.data
    assert MASK_BYTES in masked.data
    print("✓ Array mode includes the mask when supplied")

    assert provider.max_input_images == 16
    print("✓ Provider declares its input image limit")


def test_edit_response_parsing():
    """Test parsing of edit responses, including failure modes."""
    print("\n=== Testing Edit Response Parsing ===")

    provider = OpenAIProvider()

    body = json.dumps(
        {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
    ).encode("utf-8")
    assert provider.parse_edit_response(body) == PNG_BYTES
    print("✓ Edit response image data is decoded")

    try:
        provider.parse_edit_response(json.dumps({"data": []}).encode("utf-8"))
        raise AssertionError("Expected ProviderError for empty data")
    except ProviderError as e:
        assert str(e) == "No data in API response"
    print("✓ Empty data raises ProviderError")

    error = provider.format_http_error(400, "x" * 500)
    assert error.startswith("GPT-Image-1 API error 400: ")
    assert len(error) < 260  # body is truncated to 200 chars
    print("✓ HTTP errors are formatted and truncated")


def test_multipart_encoding():
    """Test the multipart encoder used by the edits endpoint."""
    print("\n=== Testing Multipart Encoding ===")

    body, boundary = create_multipart_data(
        {"prompt": "hello"}, {"image": ("image.png", PNG_BYTES, "image/png")}
    )

    assert isinstance(body, bytes)
    assert boundary in body.decode("latin-1")
    assert body.startswith(f"--{boundary}\r\n".encode())
    assert body.endswith(f"--{boundary}--\r\n".encode())
    print("✓ Boundaries open and close the body")

    # CRLF line endings matter - Windows text mode must never touch this
    assert b"\r\n" in body
    assert b"\n\n" not in body.replace(b"\r\n", b"")
    print("✓ Parts use CRLF line endings")

    assert b'Content-Disposition: form-data; name="prompt"' in body
    assert b"hello" in body
    assert b"Content-Type: image/png" in body
    assert PNG_BYTES in body
    print("✓ Text fields and file parts are encoded correctly")

    # Every call gets a fresh boundary
    _, other_boundary = create_multipart_data({}, {})
    assert boundary != other_boundary
    print("✓ Boundaries are unique per call")


def test_round_trip_with_mocked_http():
    """Test a full request/response cycle against a canned HTTP response."""
    print("\n=== Testing Round Trip (Mocked HTTP) ===")

    provider = OpenAIProvider()
    captured = {}

    def fake_urlopen(req, timeout=None):
        """Stand-in for the plugin's transport - records and replies."""
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = req.data

        class FakeResponse:
            def read(self):
                return json.dumps(
                    {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return FakeResponse()

    req = provider.build_generation_request("a cat", "1024x1024", "sk-test")
    with fake_urlopen(req, timeout=180) as response:
        image_data = provider.parse_generation_response(response.read())

    assert captured["url"] == "https://api.openai.com/v1/images/generations"
    assert captured["auth"] == "Bearer sk-test"
    assert image_data == PNG_BYTES
    print("✓ Generation round trip returns the image bytes")

    req = provider.build_edit_request(PNG_BYTES, MASK_BYTES, "sky", "1024x1024", "sk-t")
    with fake_urlopen(req, timeout=120) as response:
        image_data = provider.parse_edit_response(response.read())

    assert captured["url"] == "https://api.openai.com/v1/images/edits"
    assert image_data == PNG_BYTES
    print("✓ Edit round trip returns the image bytes")


def test_venice_capabilities():
    """Test the capability flags that change how the plugin behaves."""
    print("\n=== Testing Venice Capabilities ===")

    provider = VeniceProvider({})

    assert provider.name == "venice"
    assert provider.label == "Venice"
    assert provider.experimental is True
    print("✓ Venice is flagged experimental")

    # No Venice endpoint takes a mask, so the plugin masks locally instead
    assert provider.requires_mask is False
    assert OpenAIProvider({}).requires_mask is True
    print("✓ Venice declares it has no mask parameter")

    # Limit for multi-edit, refreshed from the catalogue by the settings dialog
    assert provider.max_input_images == VeniceProvider.DEFAULT_MAX_INPUT_IMAGES
    assert VeniceProvider({"venice": {"max_input_images": 3}}).max_input_images == 3
    assert VeniceProvider({"venice": {"max_input_images": "2"}}).max_input_images == 2
    assert (
        VeniceProvider({"venice": {"max_input_images": "nonsense"}}).max_input_images
        == VeniceProvider.DEFAULT_MAX_INPUT_IMAGES
    )
    print("✓ Input image limit reads config and survives bad values")

    assert provider.generation_models()
    assert provider.edit_models()
    print("✓ Fallback model lists are non-empty")


def test_venice_generation_request():
    """Test generation against Venice's OpenAI compatible endpoint."""
    print("\n=== Testing Venice Generation Request ===")

    provider = VeniceProvider({"venice": {"model": "qwen-image-3"}})

    assert provider.resolve_generation_size("auto") == "1536x1024"
    assert provider.resolve_generation_size("1024x1024") == "1024x1024"
    print("✓ Generation size resolution matches the plugin's shapes")

    req = provider.build_generation_request("a red dragon", "1536x1024", "vn-key")

    assert req.full_url == "https://api.venice.ai/api/v1/images/generations"
    assert req.get_header("Authorization") == "Bearer vn-key"
    assert req.get_header("Content-type") == "application/json"

    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "qwen-image-3"
    assert body["prompt"] == "a red dragon"
    assert body["size"] == "1536x1024"
    assert body["response_format"] == "b64_json"
    assert body["output_format"] == "png"
    assert body["n"] == 1
    print("✓ Generation request matches the documented parameters")

    # Default model when none is configured
    default_body = json.loads(
        VeniceProvider({}).build_generation_request("x", "1024x1024", "k").data
    )
    assert default_body["model"] == VeniceProvider.DEFAULT_MODEL
    print("✓ Falls back to the default generation model")

    # Response shape is OpenAI compatible
    body = json.dumps(
        {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
    ).encode("utf-8")
    assert provider.parse_generation_response(body) == PNG_BYTES
    print("✓ Generation response decodes like OpenAI's")


def test_venice_aspect_ratio_mapping():
    """Test that plugin shapes map onto Venice aspect ratios."""
    print("\n=== Testing Venice Aspect Ratio Mapping ===")

    # With the model's supported list known, the three plugin shapes map on
    known = VeniceProvider(
        {"venice": {"edit_aspect_ratios": ["1:1", "3:2", "2:3", "16:9"]}}
    )
    assert known._aspect_ratio("1024x1024") == "1:1"
    assert known._aspect_ratio("1536x1024") == "3:2"
    assert known._aspect_ratio("1024x1536") == "2:3"
    print("✓ The three plugin shapes map to 1:1, 3:2 and 2:3")

    # Unknown sizes have no mapping, so nothing is sent
    assert known._aspect_ratio("800x600") is None
    assert known._aspect_ratio(None) is None
    print("✓ Unmapped sizes send no aspect ratio")

    # Without catalogue data nothing is sent - Venice infers the ratio from
    # the input image, which the plugin has already padded to the target shape
    assert VeniceProvider({})._aspect_ratio("1536x1024") is None
    print("✓ Nothing is sent when the model's list is unknown")

    # A model whose list lacks the value gets nothing rather than a 400.
    # qwen-image-2-edit really does omit "auto", and three models omit it.
    narrow = VeniceProvider({"venice": {"edit_aspect_ratios": ["1:1", "9:16"]}})
    assert narrow._aspect_ratio("1536x1024") is None
    assert narrow._aspect_ratio("1024x1024") == "1:1"
    print("✓ A ratio outside the model's list is dropped")


def test_venice_edit_request():
    """Test single image editing - Venice's /image/edit."""
    print("\n=== Testing Venice Edit Request ===")

    provider = VeniceProvider(
        {
            "venice": {
                "edit_model": "gpt-image-2-edit",
                "resolution": "2K",
                "edit_resolutions": ["1K", "2K", "4K"],
                "edit_aspect_ratios": ["1:1", "3:2", "2:3"],
            }
        }
    )
    req = provider.build_edit_request(
        PNG_BYTES, MASK_BYTES, "blue sky", "1536x1024", "vn-key"
    )

    assert req.full_url == "https://api.venice.ai/api/v1/image/edit"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer vn-key"

    body = json.loads(req.data.decode("utf-8"))

    # /image/edit uses "model" - multi-edit uses "modelId"
    assert body["model"] == "gpt-image-2-edit"
    assert "modelId" not in body
    assert body["prompt"] == "blue sky"
    assert body["aspect_ratio"] == "3:2"
    assert body["resolution"] == "2K"
    assert body["output_format"] == "png"
    print("✓ Edit request uses model, aspect_ratio and resolution")

    # The image travels as a base64 string in the JSON body
    assert base64.b64decode(body["image"]) == PNG_BYTES
    print("✓ Image is base64 encoded into the JSON body")

    # Venice has no mask parameter, so the mask must not be smuggled in
    assert "mask" not in body
    assert base64.b64encode(MASK_BYTES).decode("ascii") not in json.dumps(body)
    print("✓ Mask is not sent - Venice has no mask parameter")

    # A missing mask is fine for Venice
    no_mask = json.loads(
        provider.build_edit_request(PNG_BYTES, None, "x", "1024x1024", "k").data
    )
    assert no_mask["aspect_ratio"] == "1:1"
    print("✓ Works with no mask at all")


def test_venice_sizing_is_model_specific():
    """Test that unsupported sizing parameters are never sent.

    13 of Venice's 24 edit models have no resolution tiers at all - including
    the default firered-image-edit - and sending one is a 400. Three models
    reject "auto" as an aspect ratio. Both are omitted unless the catalogue
    says the model takes them.
    """
    print("\n=== Testing Venice Model-Specific Sizing ===")

    # Default config: nothing known about the model, so nothing is sent
    default = VeniceProvider({"venice": {"edit_model": "firered-image-edit"}})
    body = json.loads(
        default.build_edit_request(PNG_BYTES, None, "an eagle", "1536x1024", "k").data
    )
    assert "resolution" not in body
    assert "aspect_ratio" not in body
    assert body["model"] == "firered-image-edit"
    assert body["prompt"] == "an eagle"
    print("✓ Nothing model-specific is sent without catalogue data")

    # A tier the model does not support is dropped rather than sent
    unsupported = VeniceProvider(
        {
            "venice": {
                "edit_model": "firered-image-edit",
                "resolution": "2K",
                "edit_resolutions": [],
            }
        }
    )
    body = json.loads(
        unsupported.build_edit_request(PNG_BYTES, None, "x", "1024x1024", "k").data
    )
    assert "resolution" not in body
    print("✓ A model with no tiers never receives a resolution")

    known = VeniceProvider(
        {
            "venice": {
                "edit_model": "grok-imagine-edit",
                "resolution": "4K",
                "edit_resolutions": ["1K", "2K"],
            }
        }
    )
    body = json.loads(
        known.build_edit_request(PNG_BYTES, None, "x", "1024x1024", "k").data
    )
    assert "resolution" not in body
    print("✓ A tier outside the model's list is dropped")

    # An aspect ratio is sent only when the model lists it
    ratios = VeniceProvider(
        {
            "venice": {
                "edit_model": "qwen-image-2-edit",
                "edit_aspect_ratios": ["1:1", "3:2", "2:3"],
            }
        }
    )
    body = json.loads(
        ratios.build_edit_request(PNG_BYTES, None, "x", "1536x1024", "k").data
    )
    assert body["aspect_ratio"] == "3:2"
    print("✓ A supported aspect ratio is sent")

    # Multi-edit follows the same rule
    multi = json.loads(
        default.build_edit_request([b"A", b"B"], None, "x", "1536x1024", "k").data
    )
    assert "aspect_ratio" not in multi
    assert "resolution" not in multi
    assert multi["modelId"] == "firered-image-edit"
    print("✓ Multi-edit omits them too")


def test_venice_multi_edit_request():
    """Test layer composite - Venice's /image/multi-edit."""
    print("\n=== Testing Venice Multi-Edit Request ===")

    provider = VeniceProvider(
        {
            "venice": {
                "edit_model": "seedream-v4-edit",
                "resolution": "1K",
                "edit_resolutions": ["1K", "2K"],
                "edit_aspect_ratios": ["1:1", "3:2", "2:3"],
            }
        }
    )
    layers = [b"\x89PNG-base", b"\x89PNG-upper-1", b"\x89PNG-upper-2"]

    req = provider.build_edit_request(layers, None, "blend these", "1024x1536", "k")

    assert req.full_url == "https://api.venice.ai/api/v1/image/multi-edit"
    body = json.loads(req.data.decode("utf-8"))

    # multi-edit uses modelId, not model
    assert body["modelId"] == "seedream-v4-edit"
    assert "model" not in body
    print("✓ Multi-edit uses modelId rather than model")

    assert len(body["images"]) == 3
    for sent, original in zip(body["images"], layers):
        assert base64.b64decode(sent) == original
    print("✓ Base image first, then the layers, all base64 encoded")

    assert body["aspect_ratio"] == "2:3"
    assert body["resolution"] == "1K"
    print("✓ Sizing parameters are carried through")


def test_venice_edit_response_parsing():
    """Test the raw bytes response, and errors arriving as JSON."""
    print("\n=== Testing Venice Edit Response Parsing ===")

    provider = VeniceProvider({})

    # Success is raw image bytes, not JSON
    assert provider.parse_edit_response(PNG_BYTES) == PNG_BYTES
    print("✓ Raw image bytes pass straight through")

    # A JSON body from a binary endpoint means failure
    try:
        provider.parse_edit_response(b'{"error": "Invalid model specified"}')
        raise AssertionError("Expected ProviderError for a JSON body")
    except ProviderError as e:
        assert "Invalid model specified" in str(e)
    print("✓ JSON error body raises ProviderError with the message")

    try:
        provider.parse_edit_response(b"")
        raise AssertionError("Expected ProviderError for an empty body")
    except ProviderError as e:
        assert "Empty response" in str(e)
    print("✓ Empty body raises ProviderError")


def test_venice_http_errors():
    """Test that Venice status codes turn into actionable messages."""
    print("\n=== Testing Venice HTTP Errors ===")

    provider = VeniceProvider({})

    message = provider.format_http_error(402, '{"error": "InsufficientBalance"}')
    assert "402" in message
    assert "insufficient balance" in message.lower()
    assert "InsufficientBalance" in message
    print("✓ 402 explains the balance problem and keeps the API detail")

    for code, expected in (
        (401, "key"),
        (429, "rate limit"),
        (503, "capacity"),
    ):
        message = provider.format_http_error(code, "{}")
        assert str(code) in message
        assert expected in message.lower()
    print("✓ 401, 429 and 503 carry a hint")

    # Venice's error field is a string, unlike OpenAI's nested object
    assert "bad prompt" in provider.format_http_error(400, '{"error": "bad prompt"}')
    assert "nested msg" in provider.format_http_error(
        400, '{"error": {"message": "nested msg"}}'
    )
    print("✓ Both string and object error shapes are read")

    # Non-JSON bodies must not blow up
    assert "gateway" in provider.format_http_error(500, "bad gateway").lower()
    assert provider.format_http_error(400, "")
    print("✓ Non-JSON and empty bodies are handled")


def test_venice_model_discovery():
    """Test the public model catalogue request and parsing."""
    print("\n=== Testing Venice Model Discovery ===")

    provider = VeniceProvider({})

    req = provider.build_models_request("generation")
    assert req.full_url == "https://api.venice.ai/api/v1/models?type=image"
    print("✓ Generation models come from type=image")

    req = provider.build_models_request("edit")
    assert req.full_url == "https://api.venice.ai/api/v1/models?type=inpaint"
    print("✓ Edit models come from type=inpaint")

    # The catalogue is public - sending a key would demand a scope an
    # inference-only key does not have
    assert req.get_header("Authorization") is None
    print("✓ No API key is sent to the catalogue")

    assert provider.build_models_request("nonsense") is None
    print("✓ Unknown model kinds return no request")

    # The docs say capabilities.maxInputImages, the live API uses
    # constraints.maxInputImages - both are read
    body = json.dumps(
        {
            "data": [
                {
                    "id": "firered-image-edit",
                    "model_spec": {
                        "constraints": {
                            "maxInputImages": 6,
                            "aspectRatios": ["1:1", "3:2"],
                            "promptCharacterLimit": 1500,
                        }
                    },
                },
                {
                    "id": "docs-shape-edit",
                    "model_spec": {"capabilities": {"maxInputImages": 4}},
                },
                {
                    "id": "luma-uni-1-edit",
                    "model_spec": {"constraints": {"combineImages": False}},
                },
                {"id": "bare-model"},
                {"model_spec": {}},
            ]
        }
    ).encode("utf-8")

    models = provider.parse_models_response(body)
    by_id = {m["id"]: m for m in models}

    assert len(models) == 4  # the entry with no id is skipped
    assert by_id["firered-image-edit"]["max_input_images"] == 6
    assert by_id["firered-image-edit"]["aspect_ratios"] == ["1:1", "3:2"]
    assert by_id["firered-image-edit"]["resolutions"] == []
    assert by_id["firered-image-edit"]["prompt_limit"] == 1500
    print("✓ Limits are read from model_spec.constraints")

    assert by_id["docs-shape-edit"]["max_input_images"] == 4
    print("✓ The documented capabilities location is read as a fallback")

    # combineImages false means the model takes a single image
    assert by_id["luma-uni-1-edit"]["max_input_images"] == 1
    print("✓ combineImages false becomes a limit of one")

    assert by_id["bare-model"]["max_input_images"] is None
    print("✓ Models declaring no limit report None")


def test_region_hint():
    """Test naming the edit region for providers that take no mask."""
    print("\n=== Testing Region Hint ===")

    # 3x3 grid, by the centre of the selection
    assert describe_selection_position((400, 400, 600, 600), 1000, 1000) == "the centre"
    assert describe_selection_position((0, 0, 200, 200), 1000, 1000) == "the top left"
    assert (
        describe_selection_position((800, 800, 1000, 1000), 1000, 1000)
        == "the bottom right"
    )
    assert describe_selection_position((400, 0, 600, 200), 1000, 1000) == "the top"
    assert describe_selection_position((0, 400, 200, 600), 1000, 1000) == "the left"
    print("✓ Selections map onto a 3x3 grid")

    # A near-full-frame selection has no meaningful position
    assert describe_selection_position((0, 0, 1000, 1000), 1000, 1000) is None
    assert describe_selection_position((10, 10, 990, 990), 1000, 1000) is None
    print("✓ Full-frame selections report no region")

    # Degenerate input must not raise
    assert describe_selection_position(None, 1000, 1000) is None
    assert describe_selection_position((0, 0, 0, 0), 1000, 1000) is None
    assert describe_selection_position((0, 0, 100, 100), 0, 0) is None
    print("✓ Degenerate bounds return None")

    # Venice names the region; OpenAI leaves the prompt alone
    venice = VeniceProvider({})
    assert (
        venice.add_region_hint("add a dwarf sitting", "the centre")
        == "add a dwarf sitting, in the centre of the image"
    )
    assert venice.add_region_hint("a bench.", "the top") == "a bench, in the top of the image"
    print("✓ Venice appends the region to the prompt")

    assert (
        OpenAIProvider({}).add_region_hint("add a dwarf sitting", "the centre")
        == "add a dwarf sitting"
    )
    print("✓ OpenAI prompts are untouched")

    # No region, empty prompt, or a prompt that already says it
    assert venice.add_region_hint("a bench", None) == "a bench"
    assert venice.add_region_hint("", "the centre") == ""
    already = "a bench in the centre"
    assert venice.add_region_hint(already, "the centre") == already
    print("✓ No hint when there is nothing to add")


def run_all_tests():
    """Run all provider tests."""
    print("Provider Abstraction Test Suite")
    print("=" * 60)

    try:
        test_provider_selection()
        test_api_key_lookup()
        test_generation_request()
        test_generation_response_parsing()
        test_edit_request_single_image()
        test_edit_request_image_array()
        test_edit_response_parsing()
        test_multipart_encoding()
        test_round_trip_with_mocked_http()
        test_venice_capabilities()
        test_venice_generation_request()
        test_venice_aspect_ratio_mapping()
        test_venice_edit_request()
        test_venice_sizing_is_model_specific()
        test_venice_multi_edit_request()
        test_venice_edit_response_parsing()
        test_venice_http_errors()
        test_venice_model_discovery()
        test_region_hint()

        print("\n" + "=" * 60)
        print("🎉 ALL PROVIDER TESTS PASSED!")
        print("✓ Provider selection is backward compatible")
        print("✓ API keys resolve from config and environment")
        print("✓ Requests match the OpenAI images API")
        print("✓ Responses decode to image bytes")
        print("✓ Failure modes raise ProviderError")
        print("✓ Venice requests match its documented endpoints")
        print("✓ Venice raw image responses and JSON errors are handled")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

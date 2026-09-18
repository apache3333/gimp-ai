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
    OpenAIProvider,
    ProviderError,
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

    provider = OpenAIProvider()

    # Config takes priority over the environment
    config = {"openai": {"api_key": "config-key"}}
    environ = {"OPENAI_API_KEY": "env-key"}
    assert provider.get_api_key(config, environ) == "config-key"
    print("✓ Config key takes priority")

    # Environment is the fallback
    assert provider.get_api_key({}, environ) == "env-key"
    assert provider.get_api_key({"openai": {}}, environ) == "env-key"
    assert provider.get_api_key({"openai": {"api_key": None}}, environ) == "env-key"
    print("✓ Environment variable is used as fallback")

    # Nothing configured
    assert provider.get_api_key({}, {}) is None
    assert provider.get_api_key(None, {}) is None
    print("✓ Missing key returns None")

    # The message names the provider and its env var
    message = provider.missing_key_message()
    assert "OpenAI" in message
    assert "OPENAI_API_KEY" in message
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

        print("\n" + "=" * 60)
        print("🎉 ALL PROVIDER TESTS PASSED!")
        print("✓ Provider selection is backward compatible")
        print("✓ API keys resolve from config and environment")
        print("✓ Requests match the OpenAI images API")
        print("✓ Responses decode to image bytes")
        print("✓ Failure modes raise ProviderError")

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

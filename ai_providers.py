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

    # Largest number of images the edit endpoint accepts in one call
    max_input_images = 1

    def get_api_key(self, config, environ):
        """Get the API key for this provider from config or environment.

        Args:
            config: Plugin config dict
            environ: Environment mapping (normally os.environ)

        Returns:
            str or None: The API key, or None if not configured
        """
        section = (config or {}).get(self.name) or {}
        api_key = section.get("api_key")
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


class OpenAIProvider(Provider):
    """OpenAI gpt-image-1 - generations and edits endpoints."""

    name = "openai"
    label = "OpenAI"

    api_key_env = "OPENAI_API_KEY"
    key_placeholder = "sk-proj-..."

    max_input_images = 16

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
        response_data = json.loads(body.decode("utf-8"))

        if "data" not in response_data or len(response_data["data"]) == 0:
            raise ProviderError("No data in API response")

        result_data = response_data["data"][0]
        if "b64_json" not in result_data:
            raise ProviderError("No image data in response")

        return base64.b64decode(result_data["b64_json"])

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
        response_json = json.loads(body.decode("utf-8"))

        if "data" not in response_json or len(response_json["data"]) == 0:
            raise ProviderError("No data in API response")

        result_data = response_json["data"][0]
        if "b64_json" not in result_data:
            raise ProviderError("No image data in response")

        return base64.b64decode(result_data["b64_json"])

    def format_http_error(self, code, body):
        return f"GPT-Image-1 API error {code}: {body[:200]}"


PROVIDERS = {
    OpenAIProvider.name: OpenAIProvider,
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
    return PROVIDERS[get_provider_name(config)]()

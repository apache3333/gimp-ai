# GIMP AI Plugin Beta

A Python plugin for GIMP 3.0.4+ that integrates AI image generation capabilities directly into GIMP. This is a **beta release** seeking testers on all platforms.

Supports **OpenAI's gpt-image-1** for inpainting and image generation, plus
**experimental [Venice.ai](https://venice.ai) support** with a choice of models.

## ✨ Features

- **🎨 AI Inpainting**: Fill selected areas with AI-generated content using text prompts and selection masks
- **🖼️ AI Image Generation**: Create new images from text descriptions as new layers
- **🔄 AI Layer Composite**: Intelligently blend AI content into existing images
- **⚙️ Easy Configuration**: Built-in settings dialog, no external config files needed

## 📋 Requirements

- **GIMP 3.0.4 or newer** (all GIMP 3.X versions from 3.0.4 onwards)
- **Internet connection** for AI API calls
- **An API key** from [OpenAI](https://platform.openai.com) or, experimentally, [Venice.ai](https://venice.ai)
- **Zero external dependencies** - uses only Python standard library + GIMP APIs

## 🚀 Installation

**Just 3 files to copy!** No external dependencies or complex setup.

### Easiest Way: Automated Installer 🎯

1. **Download** the [latest release ZIP](https://github.com/lukaso/gimp-ai/releases)
2. **Extract** the ZIP file
3. **Run** the installer: `python3 install_plugin.py`
4. **Restart GIMP** and configure your API key

Done! The installer handles everything automatically.

### Manual Install (3 Simple Steps)

1. **Download 3 files**: `gimp-ai-plugin.py`, `coordinate_utils.py` and `ai_providers.py`
2. **Create folder**: Make a `gimp-ai-plugin` folder in your GIMP plug-ins directory
3. **Copy files**: Put all three files in that folder, restart GIMP

### Need Help? 📖

**👉 [Read the Complete Installation Guide (INSTALL.md)](INSTALL.md) 👈**

The installation guide includes:

- ✅ Step-by-step instructions for beginners
- ✅ How to find your GIMP plugin directory on Windows/Mac/Linux
- ✅ Screenshots and visual examples
- ✅ How to get and configure your OpenAI API key
- ✅ Troubleshooting for "Filter → AI not found" issues

### Quick Reference

**Plugin folder location by OS** (replace `3.0` with your GIMP version):

- **Windows**: `%APPDATA%\GIMP\3.0\plug-ins\gimp-ai-plugin\`
- **macOS**: `~/Library/Application Support/GIMP/3.0/plug-ins/gimp-ai-plugin/`
- **Linux**: `~/.config/GIMP/3.0/plug-ins/gimp-ai-plugin/`

> **Note**: The automated installer will detect all compatible GIMP versions (3.0.4+) and let you choose which one to install to.

**Required folder structure:**

```
plug-ins/
└── gimp-ai-plugin/          ← Create this folder
    ├── gimp-ai-plugin.py    ← Required file #1
    ├── coordinate_utils.py  ← Required file #2
    └── ai_providers.py      ← Required file #3
```

**After copying files:**

- Linux/macOS: Run `chmod +x gimp-ai-plugin.py` in the folder
- Restart GIMP completely
- Look for `Filters → AI` in the menu

## ⚙️ Configuration

1. **Get an API key** from [platform.openai.com](https://platform.openai.com) (or [venice.ai](https://venice.ai))
2. In GIMP: open any `Filters → AI` tool and click **Settings**
3. **Choose your provider**, then **paste your API key**
4. Click Save - it's stored in GIMP's preferences automatically!

You can also supply the key through the `OPENAI_API_KEY` or `VENICE_API_KEY`
environment variable instead of saving it.

> **First time?** See [INSTALL.md](INSTALL.md) for detailed API key instructions.

### 🧪 Venice.ai (experimental)

Venice is a second provider you can select in Settings. Generation works the same as
OpenAI. Editing has two differences worth knowing before you rely on it:

- **No mask support.** Venice's API has no mask parameter of any kind, so it cannot be
  told *which* part of the image to change. The plugin still masks the result to your
  selection with a GIMP layer mask, so you only see changes inside your selection - but
  the model may have redrawn more of the surrounding area than OpenAI would.
- **Your prompt gets the region appended.** Because Venice composes for the whole frame,
  a subject that lands outside your selection is clipped by the layer mask - a figure can
  come back cut in half. So `add a dwarf sitting` is sent as
  `add a dwarf sitting, in the centre of the image`. **Give the subject room**: if the
  selection is much smaller than what you asked for, widen it or use Full Image mode.
- **Sizing is approximate.** Venice sizes output by aspect ratio and a resolution tier
  rather than exact pixels, so results are scaled to fit before being composited.

Venice's own documentation marks its edit endpoints as experimental, and so do we. Pick
your generation and edit models in Settings - the model list can be refreshed live from
Venice, and you can type any model id Venice adds later.

## 🎨 Usage

### AI Inpainting

**Fill selected areas with AI-generated content using text prompts**

1. **Open image** in GIMP
2. **Make a selection** of area to inpaint (or no selection for full image)
3. **Go to** `Filters → AI → Inpainting`
4. **Choose processing mode**:
   - **🔍 Focused (High Detail)**: Best for small edits, maximum resolution, selection required
   - **🖼️ Full Image (Consistent)**: Best for large changes, works with or without selection
5. **Enter prompt** (e.g., "blue sky with clouds", "remove the object")
6. **Result**: New layer with AI-generated content, automatically masked to selection area

**Selection Mask Behavior:**

- **Soft masks**: AI can redraw content _outside_ the selection to maintain visual coherence
- **With selection**: Final result is masked to show only within selected area, but AI considers surrounding context
- **No selection** (Full Image mode): Entire image is processed and replaced
- **Smart feathering**: Automatic edge blending for seamless integration

⚠️ **Important**: The AI model may modify areas outside your selection to create coherent results. Only the final output is masked to your selection.

### AI Image Generation

1. **Open or create** any GIMP document
2. **Go to** `Filters → AI → Image Generator`
3. **Enter prompt** (e.g., "a red dragon on mountain")
4. **New layer created** with generated image

### AI Layer Composite

**Intelligently combines multiple visible layers using AI guidance**

1. **Set up your layers**:

   - **Bottom layer** = base/background (what gets modified)
   - **Upper layers** = elements to integrate (people, objects, etc.)
   - Make sure desired layers are **visible**

2. **Go to** `Filters → AI → Layer Composite`

3. **Enter integration prompt** (e.g., "blend the person naturally into the forest scene")

4. **Choose mode**:

   - **✅ Include selection mask**: Uses selection on base layer to limit where changes occur
   - **❌ No mask**: AI can modify the entire base layer to integrate upper layers

5. **Result**: A new layer is created, taking the base layer and intelligently modifying it to incorporate all visible layers

## 🐛 Find Issues?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first
- Report at: [GitHub Issues](https://github.com/yourusername/gimp-ai/issues)

## 📚 Documentation

- **[INSTALL.md](INSTALL.md)** - Complete installation guide for users
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[CHANGELOG.md](CHANGELOG.md)** - What's new and known issues
- **[TODO.md](TODO.md)** - Development roadmap
- **[RELEASE.md](RELEASE.md)** - Release process for maintainers

## 🔧 For Developers

### Creating Releases

Releases are automated via GitHub Actions. See **[RELEASE.md](RELEASE.md)** for details.

**Quick overview:**

- Label PRs with `major`, `minor`, or `patch` for version bumps
- Merge to `main` triggers automated release creation
- The workflow builds the package and creates a GitHub release

### Tools

- **`build_release.py`** - Creates distributable ZIP packages (used by workflow)
- **`tools/bump_version.py`** - Bumps version in `gimp-ai-plugin.py` (used by workflow)

## ⚖️ License

MIT License - see [LICENSE](LICENSE) file for details

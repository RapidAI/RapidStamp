# Bundled CJK subset

`cjk-subset.ttf` is a glyph subset of **Droid Sans Fallback** (copyright 2006–2010 Google Corp.), produced so this repository can render synthetic seal characters without shipping the full font.

Droid Sans Fallback is licensed under the Apache License, Version 2.0. See the repository `LICENSE` file. "Droid" is a trademark of Google Corp.; this project does not use that name for its product.

The subset was generated with `fontTools.subset` from the Debian `fonts-droid-fallback` package and contains only the characters listed in `rapidstamp.charset.CHARSET`.

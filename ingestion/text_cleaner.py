import re


class TextCleaner:
    """Conservative text cleaning utility for legal document processing."""

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # Remove null bytes or control chars except newlines/tabs
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Fix hyphenated words broken across lines: e.g., "con-\nstitution" -> "constitution"
        cleaned = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned)

        # Replace multiple spaces/tabs with single space
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)

        # Normalize excessive vertical spacing (more than 2 newlines to 2 newlines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Strip leading and trailing whitespace per line
        lines = [line.strip() for line in cleaned.split('\n')]
        cleaned = '\n'.join(lines).strip()

        return cleaned

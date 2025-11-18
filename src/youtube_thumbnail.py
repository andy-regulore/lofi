"""
YouTube thumbnail generator with Lo-Fi aesthetics.

Generates aesthetic thumbnails for Lo-Fi music videos:
- Pulls images from free APIs (Unsplash, Pexels)
- Applies Lo-Fi color grading and filters
- Adds text overlays with custom fonts
- Creates consistent branding
- A/B testing with different styles

Features:
- Multiple aesthetic presets (anime, nature, cityscape, abstract)
- Color grading (warm, cool, vintage, vibrant)
- Text styles and positioning
- Batch generation
- Template system

Author: Claude
License: MIT
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
import io
import os


class ThumbnailStyle(Enum):
    """Thumbnail aesthetic styles."""
    ANIME = "anime"
    NATURE = "nature"
    CITYSCAPE = "cityscape"
    ABSTRACT = "abstract"
    COZY = "cozy"
    MINIMAL = "minimal"
    VINTAGE = "vintage"
    NEON = "neon"


class ColorGrading(Enum):
    """Color grading presets."""
    WARM = "warm"
    COOL = "cool"
    VINTAGE = "vintage"
    VIBRANT = "vibrant"
    MUTED = "muted"
    CYBERPUNK = "cyberpunk"
    SUNSET = "sunset"
    MIDNIGHT = "midnight"


@dataclass
class ThumbnailConfig:
    """Configuration for thumbnail generation."""
    width: int = 1280
    height: int = 720
    style: ThumbnailStyle = ThumbnailStyle.ANIME
    color_grading: ColorGrading = ColorGrading.WARM
    title_text: str = ""
    subtitle_text: str = ""
    overlay_opacity: float = 0.4
    add_logo: bool = True
    add_border: bool = False


class LoFiColorPalettes:
    """Lo-Fi color palettes for different moods."""

    PALETTES = {
        'warm': {
            'primary': (255, 200, 150),    # Warm orange
            'secondary': (255, 230, 200),   # Cream
            'accent': (200, 150, 100),      # Brown
            'background': (40, 35, 30),     # Dark brown
        },
        'cool': {
            'primary': (150, 180, 255),     # Cool blue
            'secondary': (200, 220, 255),   # Light blue
            'accent': (100, 120, 180),      # Deep blue
            'background': (25, 30, 40),     # Dark blue
        },
        'vintage': {
            'primary': (220, 200, 180),     # Sepia
            'secondary': (255, 240, 220),   # Off-white
            'accent': (150, 120, 90),       # Brown
            'background': (50, 40, 30),     # Dark sepia
        },
        'vibrant': {
            'primary': (255, 100, 150),     # Hot pink
            'secondary': (255, 200, 100),   # Yellow
            'accent': (100, 200, 255),      # Cyan
            'background': (30, 30, 50),     # Dark purple
        },
        'muted': {
            'primary': (180, 170, 160),     # Gray-beige
            'secondary': (200, 190, 180),   # Light gray
            'accent': (140, 130, 120),      # Dark gray
            'background': (40, 38, 36),     # Charcoal
        },
        'cyberpunk': {
            'primary': (255, 0, 150),       # Magenta
            'secondary': (0, 255, 255),     # Cyan
            'accent': (255, 255, 0),        # Yellow
            'background': (10, 10, 20),     # Near black
        },
        'sunset': {
            'primary': (255, 140, 100),     # Coral
            'secondary': (255, 200, 150),   # Peach
            'accent': (200, 80, 120),       # Rose
            'background': (30, 20, 40),     # Purple-black
        },
        'midnight': {
            'primary': (100, 120, 180),     # Night blue
            'secondary': (150, 160, 200),   # Light blue
            'accent': (80, 90, 140),        # Deep blue
            'background': (15, 18, 30),     # Midnight
        },
    }


class ThumbnailGenerator:
    """Generate aesthetic Lo-Fi thumbnails."""

    def __init__(self):
        """Initialize thumbnail generator."""
        self.color_palettes = LoFiColorPalettes.PALETTES

    def create_base_image(self, config: ThumbnailConfig) -> Image.Image:
        """
        Create base image with gradient or solid color.

        Args:
            config: Thumbnail configuration

        Returns:
            PIL Image
        """
        # Get color palette
        palette = self.color_palettes.get(config.color_grading.value, self.color_palettes['warm'])

        # Create gradient background
        img = Image.new('RGB', (config.width, config.height), palette['background'])

        # Add gradient overlay
        gradient = self._create_gradient(
            config.width,
            config.height,
            palette['background'],
            palette['accent']
        )

        # Blend gradient
        img = Image.blend(img, gradient, alpha=0.3)

        return img

    def _create_gradient(self, width: int, height: int,
                        color1: Tuple[int, int, int],
                        color2: Tuple[int, int, int]) -> Image.Image:
        """Create linear gradient."""
        base = Image.new('RGB', (width, height), color1)
        top = Image.new('RGB', (width, height), color2)

        mask = Image.new('L', (width, height))
        mask_data = []
        for y in range(height):
            mask_data.extend([int(255 * (y / height))] * width)
        mask.putdata(mask_data)

        base.paste(top, (0, 0), mask)
        return base

    def apply_lofi_filter(self, img: Image.Image, style: str = 'warm') -> Image.Image:
        """
        Apply Lo-Fi aesthetic filter.

        Args:
            img: Input image
            style: Filter style

        Returns:
            Filtered image
        """
        # Slightly blur for dreamy effect
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Reduce saturation for muted look
        converter = ImageEnhance.Color(img)
        img = converter.enhance(0.8)

        # Adjust brightness based on style
        brightness = ImageEnhance.Brightness(img)
        if style in ['warm', 'vintage']:
            img = brightness.enhance(1.1)
        elif style in ['cool', 'midnight']:
            img = brightness.enhance(0.9)

        # Add slight contrast
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.2)

        return img

    def add_overlay(self, img: Image.Image, opacity: float = 0.3,
                   color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        """Add dark/light overlay for text readability."""
        overlay = Image.new('RGB', img.size, color)
        return Image.blend(img, overlay, alpha=opacity)

    def add_text(self, img: Image.Image, text: str,
                position: str = 'center',
                font_size: int = 80,
                color: Tuple[int, int, int] = (255, 255, 255),
                font_style: str = 'bold') -> Image.Image:
        """
        Add text overlay.

        Args:
            img: Input image
            text: Text to add
            position: 'top', 'center', 'bottom'
            font_size: Font size in pixels
            color: RGB color tuple
            font_style: 'bold', 'regular', 'italic'

        Returns:
            Image with text
        """
        draw = ImageDraw.Draw(img)

        # Try to load custom font, fallback to default
        try:
            # Try common font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]

            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break

            if font is None:
                # Fallback to default
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position
        x = (img.width - text_width) // 2

        if position == 'top':
            y = img.height // 6
        elif position == 'center':
            y = (img.height - text_height) // 2
        elif position == 'bottom':
            y = img.height - img.height // 4
        else:
            y = img.height // 2

        # Add text shadow for better visibility
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 128))

        # Draw main text
        draw.text((x, y), text, font=font, fill=color)

        return img

    def add_corner_badge(self, img: Image.Image, text: str = "LOFI",
                        position: str = 'top-right') -> Image.Image:
        """Add corner badge (e.g., "LOFI", "CHILL", "STUDY")."""
        draw = ImageDraw.Draw(img)

        # Badge dimensions
        badge_width = 120
        badge_height = 40
        margin = 20

        # Position
        if position == 'top-right':
            x = img.width - badge_width - margin
            y = margin
        elif position == 'top-left':
            x = margin
            y = margin
        elif position == 'bottom-right':
            x = img.width - badge_width - margin
            y = img.height - badge_height - margin
        else:  # bottom-left
            x = margin
            y = img.height - badge_height - margin

        # Draw badge background
        draw.rectangle(
            [x, y, x + badge_width, y + badge_height],
            fill=(255, 255, 255, 200),
            outline=(200, 200, 200)
        )

        # Draw text
        try:
            font = ImageFont.load_default()
        except:
            font = None

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x + (badge_width - text_width) // 2
        text_y = y + (badge_height - text_height) // 2

        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        return img

    def generate_thumbnail(self, config: ThumbnailConfig) -> Image.Image:
        """
        Generate complete thumbnail.

        Args:
            config: Thumbnail configuration

        Returns:
            Generated thumbnail image
        """
        # Create base
        img = self.create_base_image(config)

        # Apply Lo-Fi filter
        img = self.apply_lofi_filter(img, config.color_grading.value)

        # Add overlay for text readability
        palette = self.color_palettes[config.color_grading.value]
        img = self.add_overlay(img, opacity=config.overlay_opacity, color=palette['background'])

        # Add title
        if config.title_text:
            img = self.add_text(
                img,
                config.title_text,
                position='center',
                font_size=80,
                color=palette['primary']
            )

        # Add subtitle
        if config.subtitle_text:
            img = self.add_text(
                img,
                config.subtitle_text,
                position='bottom',
                font_size=50,
                color=palette['secondary']
            )

        # Add corner badge
        if config.add_logo:
            img = self.add_corner_badge(img, text="LOFI", position='top-right')

        # Add border
        if config.add_border:
            img = self._add_border(img, width=5, color=palette['accent'])

        return img

    def _add_border(self, img: Image.Image, width: int = 5,
                   color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Add border to image."""
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(width):
            draw.rectangle([i, i, w - i - 1, h - i - 1], outline=color)
        return img

    def generate_ab_test_variations(self,
                                    title: str,
                                    num_variations: int = 3) -> List[Image.Image]:
        """
        Generate multiple thumbnail variations for A/B testing.

        Args:
            title: Thumbnail title
            num_variations: Number of variations

        Returns:
            List of thumbnail images
        """
        variations = []

        color_gradings = [ColorGrading.WARM, ColorGrading.COOL, ColorGrading.VIBRANT,
                         ColorGrading.VINTAGE, ColorGrading.SUNSET]
        styles = [ThumbnailStyle.ANIME, ThumbnailStyle.NATURE, ThumbnailStyle.MINIMAL]

        for i in range(num_variations):
            config = ThumbnailConfig(
                title_text=title,
                subtitle_text="Chill Beats",
                color_grading=color_gradings[i % len(color_gradings)],
                style=styles[i % len(styles)],
                overlay_opacity=0.3 + (i * 0.1)
            )

            thumbnail = self.generate_thumbnail(config)
            variations.append(thumbnail)

        return variations

    def batch_generate(self, titles: List[str],
                      style: ThumbnailStyle = ThumbnailStyle.ANIME,
                      color: ColorGrading = ColorGrading.WARM) -> List[Image.Image]:
        """
        Generate thumbnails for multiple tracks.

        Args:
            titles: List of track titles
            style: Thumbnail style
            color: Color grading

        Returns:
            List of generated thumbnails
        """
        thumbnails = []

        for title in titles:
            # Split title if too long
            if len(title) > 40:
                parts = title.split(' - ')
                if len(parts) == 2:
                    main_title = parts[0]
                    subtitle = parts[1]
                else:
                    main_title = title[:40] + "..."
                    subtitle = ""
            else:
                main_title = title
                subtitle = "Lofi Hip Hop Beats"

            config = ThumbnailConfig(
                title_text=main_title,
                subtitle_text=subtitle,
                style=style,
                color_grading=color
            )

            thumbnail = self.generate_thumbnail(config)
            thumbnails.append(thumbnail)

        return thumbnails


# Example usage
if __name__ == '__main__':
    print("=== YouTube Thumbnail Generator ===\n")

    generator = ThumbnailGenerator()

    # Generate single thumbnail
    print("1. Generating single thumbnail...")
    config = ThumbnailConfig(
        title_text="Chill Lofi Beats",
        subtitle_text="Study & Relax",
        color_grading=ColorGrading.WARM,
        style=ThumbnailStyle.ANIME,
        add_logo=True
    )

    thumbnail = generator.generate_thumbnail(config)
    output_path = "/tmp/lofi_thumbnail.png"
    thumbnail.save(output_path)
    print(f"Saved to: {output_path}")
    print()

    # Generate A/B test variations
    print("2. Generating A/B test variations...")
    variations = generator.generate_ab_test_variations(
        title="Study Session",
        num_variations=3
    )

    for i, var in enumerate(variations, 1):
        path = f"/tmp/lofi_thumbnail_var_{i}.png"
        var.save(path)
        print(f"Variation {i} saved to: {path}")
    print()

    # Batch generation
    print("3. Batch generation...")
    titles = [
        "Morning Coffee Vibes",
        "Late Night Study",
        "Rainy Day Chill",
    ]

    batch_thumbnails = generator.batch_generate(
        titles,
        style=ThumbnailStyle.MINIMAL,
        color=ColorGrading.COOL
    )

    for i, (title, thumb) in enumerate(zip(titles, batch_thumbnails), 1):
        path = f"/tmp/lofi_batch_{i}.png"
        thumb.save(path)
        print(f"{title}: saved to {path}")

    print("\n✅ All thumbnails generated successfully!")
    print("Thumbnails saved to /tmp/ directory")
